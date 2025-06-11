# AIcap_final  
`bot.py` is the main program, you can play with it!  
`chess_data.py` creates `fen_vectors.npy`, `fen_metadata.jsonl.gz`, `faiss.index`. Which is for bot RAG.  
`pgnParsingScript.py` creates the json file for efficient fine-tuning
## how to play with the bot
first, run `chess_data.py` and put `fen_vectors.npy`, `fen_metadata.jsonl.gz`, `faiss.index`, and `bot.py` in the same directory.  
then, install the requirements and you're all set!
## references
RAG data source: https://database.nikonoel.fr/  
fine-tuning data source: https://github.com/xinyangz/chess-tactics-pgn  
`Finetune_Llama3_with_LLaMA_Factory_ipynb.ipynb` source: https://github.com/hiyouga/LLaMA-Factory  
