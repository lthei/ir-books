from pathlib import Path

# project root is the directory that contains this file
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

BOOKS_JSON     = DATA_DIR / "goodreads_books.json"   # raw book data from fetch.py
INDEX_PICKLE   = DATA_DIR / "inverted_index.pkl"     # inverted index + document corpus
EMBEDDINGS_NPY = DATA_DIR / "doc_embeddings.npy"     # sentence-transformer embeddings cache
BOOKS_DB       = DATA_DIR / "books.db"               # SQLite metadata database