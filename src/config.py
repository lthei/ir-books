from pathlib import Path

ROOT = Path(__file__).parent.parent  # goes up from src/ to ir-books/
DATA_DIR = ROOT / "data"

BOOKS_JSON = DATA_DIR / "goodreads_books.json"
EMBEDDINGS_NPY = DATA_DIR / "doc_embeddings.npy"