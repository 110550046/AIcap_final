import chess
import chess.pgn
import faiss
import numpy as np
import gzip
from tqdm import tqdm
import json

def fen_to_vector(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    piece_map = board.piece_map()

    piece_encoding = {
        'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
        'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6
    }

    vec = np.zeros(64)
    for square, piece in piece_map.items():
        vec[square] = piece_encoding[str(piece)]

    return vec.astype("float32")

def extract_and_save(pgn_path, max_games=100000, out_vector="fen_vectors.npy", out_meta="fen_metadata.jsonl.gz"):
    vectors = []
    meta_id = 0

    with open(pgn_path, "r", encoding="utf-8") as f_in, gzip.open(out_meta, "wt", encoding="utf-8") as f_meta:
        # 用 tqdm 包住一個 range，模擬處理 max_games 次
        for _ in tqdm(range(max_games), desc="處理 PGN 對局"):
            game = chess.pgn.read_game(f_in)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                uci = move.uci()
                vec = fen_to_vector(fen)
                vectors.append(vec)
                json.dump({"id": meta_id, "fen": fen, "move": uci}, f_meta)
                f_meta.write("\n")
                meta_id += 1
                board.push(move)

    # 儲存向量
    np.save(out_vector, np.array(vectors, dtype=np.float32))

def query_similar_moves(board, index_file="faiss.index", meta_file="fen_metadata.jsonl.gz", top_k=5):
    index = faiss.read_index(index_file)
    vec = fen_to_vector(board.fen()).reshape(1, -1)
    D, I = index.search(vec, top_k)

    # 讀取 metadata（這邊只載入對應行，避免記憶體爆）
    results = []
    with gzip.open(meta_file, "rt", encoding="utf-8") as f_meta:
        lines = f_meta.readlines()

    for idx in I[0]:
        item = json.loads(lines[idx])
        try:
            move = chess.Move.from_uci(item["move"])
            if move in board.legal_moves:
                results.append(item)
        except:
            continue

    return results

def build_faiss_index(vector_file="fen_vectors.npy", index_file="faiss.index"):
    vectors = np.load(vector_file)
    index = faiss.IndexFlatL2(64)  # 使用 L2 距離
    index.add(vectors)
    faiss.write_index(index, index_file)


# 1. 轉換資料集
extract_and_save("lichess_elite_2020-05.pgn", max_games=10000)

# 2. 建立 FAISS 索引
build_faiss_index()

# 3. 查詢某局面
board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
similar = query_similar_moves(board)
for item in similar:
    print(f"FEN: {item['fen']} → move: {item['move']}")
