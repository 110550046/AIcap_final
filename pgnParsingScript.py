import chess
import re
import json

def parse_tactics_pgn(pgn_text):
    pgn_text = re.sub(r'\]\[', ']\n[', pgn_text)
    games = re.split(r'\n\n(?=\[Event)', pgn_text.strip())
    dataset = []

    for game in games:
        fen_match = re.search(r'\[FEN\s+"([^"]+)"\]', game)
        moves_match = re.search(r'\n(?:\d+)[\.\.]*\s*(.+?)(?:\n\n|\Z)', game, re.DOTALL)

        if not fen_match or not moves_match:
            continue

        fen = fen_match.group(1).strip()
        moves_raw = moves_match.group(1).strip()
        moves_san = [m.strip() for m in re.split(r'\s+', moves_raw) if m.strip()]

        if len(moves_san) < 2:
            continue

        board = chess.Board(fen)

        try:
            board.push_san(moves_san[0])
        except Exception:
            continue

        fen_after_first = board.fen()
        legal_moves = [move.uci() for move in board.legal_moves]

        try:
            target_uci = board.parse_san(moves_san[1]).uci()
        except Exception:
            continue

        dataset.append({
            "instruction": f"You are a chess grandmaster.\nGiven the current board state and legal moves, suggest the best move.\nBoard (FEN): {fen_after_first}\nLegal moves: {legal_moves}\nChoose the best move from the Legal moves list.",
            "input": "",
            "output": f"best move: {target_uci}"
        })

    return dataset

# 使用範例
with open("tactics.pgn", "r", encoding="utf-8") as f:
    pgn_data = f.read()

dataset = parse_tactics_pgn(pgn_data)

# 儲存為 JSON
with open("llama_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)