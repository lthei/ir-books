"""
Run this script once after fetch.py to build all search indexes.
It assumes goodreads_books.json already exists in the data/ folder.

Usage (from the src/ directory):
    python setup.py
"""

from rank_bm25 import BM25Okapi
from index import load_books, build_corpus, build_inverted_index, save_index, build_sqlite_db
from preprocess import simple_tokenize
from config import EMBEDDINGS_NPY
from sentence_transformers import SentenceTransformer
import numpy as np


def main():
    # load books and build the document corpus
    books = load_books()
    docs = build_corpus(books)

    # build and save the inverted index and SQLite metadata database
    inverted_index, document_corpus = build_inverted_index(docs)
    save_index(inverted_index, document_corpus)
    build_sqlite_db(books)

    # encode all documents and save the embeddings cache
    # this is the slowest step — subsequent runs load from cache instead
    print("Encoding documents (this may take a while)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_texts = [doc["text"] for doc in docs]
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, doc_embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_NPY}")

    # quick BM25 sanity check to confirm everything is working
    print("\nSanity check — top 3 results for 'dystopian society future':")
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    results = bm25.get_top_n(simple_tokenize("dystopian society future"), docs, n=3)
    for i, res in enumerate(results, start=1):
        print(f"  Rank {i}: {res['title']}")

    print("\nSetup complete. You can now run search.py and evaluate.py.")


if __name__ == "__main__":
    main()
