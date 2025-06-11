import chess
import chess.pgn
import time
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gzip
import json
import numpy as np
from collections import Counter
import os

# 设置环境变量以解决 OpenMP 运行时冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ChessBot:
    def __init__(self):
        self.board = chess.Board()
        self.last_move = None  # ⬅️ 追踪最後一步移動

        # Hugging Face 模型初始化
        model_id = "GIBAA/1b_chess"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        self.faiss_index = faiss.read_index("faiss.index")
        self.metadata = self.load_metadata("fen_metadata.jsonl.gz")
        
        # Minimax 相關參數
        self.max_depth = 3  # 搜尋深度，可以調整
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
        }

    def load_metadata(self, path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
        
    def fen_to_vector(self, fen: str):
        board = chess.Board(fen)
        piece_map = board.piece_map()
        encoding = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6}
        vec = np.zeros(64, dtype=np.float32)
        for square, piece in piece_map.items():
            vec[square] = encoding[str(piece)]
        return vec


    def evaluate_board(self, board):
        """評估棋盤位置，正值對白方有利，負值對黑方有利"""
        if board.is_checkmate():
            return -9999 if board.turn else 9999
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # 計算子力價值
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        
        # 位置獎勵
        score += self.evaluate_position(board)
        
        # 機動性評估
        score += self.evaluate_mobility(board)

        score += self.evaluate_development(board)

        return score

    def evaluate_development(self, board):
        """鼓勵開展主力子力與入堡，避免王走出入堡以外的步驟"""
        score = 0
        back_rank = 0 if board.turn == chess.WHITE else 7
        color = board.turn

        undeveloped_penalty = 40
        castling_penalty = 80
        king_moved_penalty = 120

        # 未開展的皇后、車、主教、馬
        initial_squares = {
            chess.WHITE: {
                'Q': [chess.D1],
                'R': [chess.A1, chess.H1],
                'B': [chess.C1, chess.F1],
                'N': [chess.B1, chess.G1],
            },
            chess.BLACK: {
                'q': [chess.D8],
                'r': [chess.A8, chess.H8],
                'b': [chess.C8, chess.F8],
                'n': [chess.B8, chess.G8],
            },
        }

        for piece_symbol, squares in initial_squares[color].items():
            for square in squares:
                piece = board.piece_at(square)
                if piece and piece.symbol() == piece_symbol:
                    score -= undeveloped_penalty

        # 國王是否已經入堡
        king_square = board.king(color)
        has_castled = (
            (color == chess.WHITE and king_square == chess.G1 or king_square == chess.C1) or
            (color == chess.BLACK and king_square == chess.G8 or king_square == chess.C8)
        )
        if not has_castled:
            # 如果王離開了初始格但沒有入堡，重罰
            if (color == chess.WHITE and king_square != chess.E1) or (color == chess.BLACK and king_square != chess.E8):
                score -= king_moved_penalty
            else:
                score -= castling_penalty  # 還沒入堡但在原地，輕微懲罰

        return score


    def evaluate_position(self, board):
        """評估棋子位置"""
        # 中心控制獎勵
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        score = 0
        
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 30
                else:
                    score -= 30
        
        # 王的安全性
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        
        if white_king and board.is_attacked_by(chess.BLACK, white_king):
            score -= 50
        if black_king and board.is_attacked_by(chess.WHITE, black_king):
            score += 50
        
        return score

    def evaluate_mobility(self, board):
        """評估機動性（可走步數）"""
        white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        board.turn = not board.turn
        black_mobility = len(list(board.legal_moves))
        board.turn = not board.turn
        
        return (white_mobility - black_mobility) * 10

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax 演算法實現，使用 alpha-beta 剪枝"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta 剪枝
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta 剪枝
            return min_eval

    def get_minimax_move(self):
        """使用 minimax 演算法獲取前10個最佳移動，並過濾掉與最佳移動評分相差過大的移動"""
        moves_with_scores = []
        
        print(f"🧠 Minimax 搜尋深度：{self.max_depth}")
        
        for move in self.board.legal_moves:
            self.board.push(move)
            if self.board.turn == chess.BLACK:  # 剛走完白棋
                score = self.minimax(self.board, self.max_depth - 1, float('-inf'), float('inf'), False)
            else:  # 剛走完黑棋
                score = self.minimax(self.board, self.max_depth - 1, float('-inf'), float('inf'), True)
            self.board.pop()
            
            moves_with_scores.append((move, score))
        
        # 根據當前玩家排序
        if self.board.turn == chess.WHITE:
            moves_with_scores.sort(key=lambda x: x[1], reverse=True)  # 白方要最大化分數
        else:
            moves_with_scores.sort(key=lambda x: x[1])  # 黑方要最小化分數
        
        # 檢查合法移動數量
        num_legal_moves = len(moves_with_scores)
        
        # 根據合法移動數量決定返回數量
        if num_legal_moves <= 5:
            # 1-5個移動時，返回全部
            num_to_return = num_legal_moves
        elif num_legal_moves <= 8:
            # 6-8個移動時，返回5個
            num_to_return = 5
        elif num_legal_moves <= 10:
            # 9-10個移動時，返回6個
            num_to_return = 6
        elif num_legal_moves <= 13:
            # 11-13個移動時，返回7個
            num_to_return = 7
        else:
            # 14個以上移動時，返回8個
            num_to_return = 8
            
        top_moves = moves_with_scores[:num_to_return]
        print(f"📊 合法移動數量：{num_legal_moves}，返回前{num_to_return}個最佳移動")
        
        # 過濾掉與最佳移動評分相差超過99的移動
        if top_moves:
            best_score = top_moves[0][1]
            filtered_moves_with_scores = [(move, score) for move, score in top_moves 
                                        if abs(score - best_score) <= 99]
            
            print(f"🎯 Minimax 最佳選擇（已過濾{len(filtered_moves_with_scores)}個）：")
            for move, score in filtered_moves_with_scores:
                print(f"移動：{move.uci()}，評分：{score}（與最佳評分相差：{abs(score - best_score)}）")
            
            return [move for move, _ in filtered_moves_with_scores]
        
        return []

    def get_same_moves(self):
        current_fen = self.board.fen()

        # 🔍 精確匹配 FEN
        exact_matches = [item["move"] for item in self.metadata if item.get("fen") == current_fen]

        if exact_matches:
            return exact_matches

    def get_similar_moves(self, top_k=5):   
        vec = self.fen_to_vector(self.board.fen()).reshape(1, -1)
        D, I = self.faiss_index.search(vec, top_k)
        legal_moves = [m.uci() for m in self.board.legal_moves]

        suggestions = []
        for idx in I[0]:
            try:
                item = self.metadata[idx]
                move = chess.Move.from_uci(item["move"])
                if move.uci() in legal_moves:
                    suggestions.append(item["move"])
            except:
                continue
        return suggestions

    def get_bot_move(self):
        # Step 1: 搜尋完全一致的局面
        print("搜尋一致的局面...")
        same_moves = self.get_same_moves()
        same_moves_counter = Counter(same_moves)
        if same_moves_counter:
            print(f"✅ 找到一致的局面，使用該局面的最常見走法")
            most_common_moves = [move for move, _ in same_moves_counter.most_common(1)]
            return chess.Move.from_uci(most_common_moves[0])
        
        # Step 2: 搜尋相似的局面
        print("⚠️ 找不到一致的局面，進行 Minimax 評估...")
        top_moves = self.get_minimax_move()
        print(f"Minimax 評估最佳走法：{[move.uci() for move in top_moves]}，共{len(top_moves)}個")
        if len(top_moves) == 1:
            print(f"✅ 執行唯一最佳走法")
            return top_moves[0]

        print("搜尋相似的局面...")
        suggested_moves = self.get_similar_moves()
        similar_moves_counter = Counter(suggested_moves)
        if similar_moves_counter:
            most_common_3_moves = [move for move, _ in similar_moves_counter.most_common(3)]
            print(f"✅ 找到相似的局面，相似局面的3種最常見走法：{most_common_3_moves}")
                
            # 檢查最常見的3個移動是否在 top_moves 中
            for common_move in most_common_3_moves:
                if common_move in [move.uci() for move in top_moves]:
                    move = chess.Move.from_uci(common_move)
                    print(f"✅ FAISS 建議走法：{common_move} 在評估最佳走法中")
                    return move
            print(f"⚠️ 相似局面的3種最常見走法 {most_common_3_moves} 都不在評估最佳走法 {[move.uci() for move in top_moves]} 中，使用 LLM fallback")
        else:
            print("⚠️ 未找到相似的局面，使用 LLM fallback")

        # Step 2: LLM fallback
        prompt = f"You are a chess grandmaster.\nGiven the current board state and legal moves, suggest the best move.\nBoard (FEN): {self.board.fen()}\nLegal moves: {[move.uci() for move in top_moves]}\nChoose the best move from the Legal moves list."
        print(f"\033[95m{prompt}\033[0m")
        outputs = self.pipeline(prompt, max_new_tokens=100, temperature=0.3, return_full_text=False)
        full_response = outputs[0]["generated_text"].strip()
        print(f"\033[93m{full_response}\033[0m")
        
        import re
        moves = re.findall(r'\b[a-h][1-8][a-h][1-8][qrbn]?\b', full_response)
        if moves:
            for move_text in moves:
                try:
                    move = chess.Move.from_uci(move_text)
                    if move in self.board.legal_moves:
                        print(f"✅ LLM 選擇走法：{move.uci()}")
                        return move
                except:
                    continue

        import random
        if top_moves:
            fallback_move = random.choice(top_moves)
            print(f"⚠️ 所有方法都失敗，隨機選擇最佳走法：{fallback_move.uci()}")
        else:
            fallback_move = random.choice(list(self.board.legal_moves))
            print(f"⚠️ 所有方法都失敗且無最佳走法，選擇隨機走法：{fallback_move.uci()}")
        return fallback_move


    def print_board_with_index(self):
        # ANSI 顏色
        YELLOW_BG = '\033[43m'
        PURPLE = '\033[95m'
        YELLOW = '\033[93m'
        RESET = '\033[0m'

        # 棋子符號映射
        PIECE_SYMBOLS = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }

        print(f"{YELLOW}   A B C D E F G H{RESET}\n")
        board_str = str(self.board)
        rows = board_str.split('\n')

        # 將棋盤轉為列表形式以便修改顏色
        board_matrix = [list(row.replace(" ", "")) for row in rows]

        # 如果有上一步的移動，標記起點和終點格子
        highlight_squares = set()
        if self.last_move:
            from_square = self.last_move.from_square
            to_square = self.last_move.to_square
            highlight_squares.add(from_square)
            highlight_squares.add(to_square)

        for i in range(8):
            print(f"{YELLOW}{8 - i}{RESET}  ", end="")
            for j in range(8):
                square_index = chess.square(j, 7 - i)
                piece = board_matrix[i][j]
                # 將棋子轉換為 Unicode 符號
                piece_symbol = PIECE_SYMBOLS.get(piece, piece)
                color_piece = f"{PURPLE}{piece_symbol}{RESET}" if piece.islower() else piece_symbol
                if square_index in highlight_squares:
                    print(f"{YELLOW_BG}{color_piece}{RESET} ", end="")
                else:
                    print(f"{color_piece} ", end="")
            print()

    def play_game(self):
        print("歡迎來到國際象棋對戰！")
        print("請選擇你要執的棋子：")
        print("1. 白方（先手）")
        print("2. 黑方（後手）")
        
        while True:
            choice = input("請輸入你的選擇（1或2）：")
            if choice in ['1', '2']:
                player_is_white = choice == '1'
                break
            print("無效的選擇，請輸入1或2")
        
        print("\n輸入格式：起始格子到目標格子（例如：e2e4）")
        print("輸入 'quit' 離開遊戲")

        while not self.board.is_game_over():
            print("\n目前棋盤：")
            self.print_board_with_index()

            if (self.board.turn == chess.WHITE) == player_is_white:
                while True:
                    move_uci = input("\n請輸入你的移動： ")
                    if move_uci.lower() == 'quit':
                        return
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in self.board.legal_moves:
                            self.board.push(move)
                            self.last_move = move  # 記錄移動
                            break
                        else:
                            print("非法移動，請再試一次。")
                    except:
                        print("輸入格式錯誤，請使用UCI格式（如：e2e4）")
            else:
                print("\nAI 思考中...")
                time.sleep(1)
                move = self.get_bot_move()
                print(f"AI移動：{move.uci()}")
                self.board.push(move)
                self.last_move = move  # 記錄AI移動

        print("\n遊戲結束！")
        print(self.board)
        if self.board.is_checkmate():
            winner = "白方" if self.board.turn == chess.BLACK else "黑方"
            print(f"{winner}獲勝！")
        elif self.board.is_stalemate():
            print("和局！")
        elif self.board.is_insufficient_material():
            print("因子力不足，和局！")

if __name__ == "__main__":
    bot = ChessBot()
    bot.play_game()