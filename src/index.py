import json
import pickle
import sqlite3
from collections import defaultdict

from preprocess import simple_tokenize
from config import BOOKS_JSON, INDEX_PICKLE, BOOKS_DB


def load_books(path=BOOKS_JSON):
    """Load the book list from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(books):
    """
    Build a list of document dicts from the raw book list.

    The 'text' field is used for indexing and ranking only — it intentionally
    excludes 'year' so publication date does not influence relevance scores.
    Genre is repeated twice to give it more weight relative to the description.
    Display fields (title, authors, year, genres) are kept separately so
    results can be presented with full metadata.
    """
    docs = []
    for b in books:
        docs.append({
            "id":          b["id"],
            "title":       b["title"],
            "description": b["description"],
            "authors":     b["authors"],
            "year":        b["year"],
            "genres":      b["genres"],
            # genre is doubled to boost its weight in BM25 scoring
            "text": (
                b["title"] + " " +
                " ".join(b["authors"]) + " " +
                " ".join(b["genres"]) + " " +
                " ".join(b["genres"]) + " " +
                b["description"]
            ),
        })
    return docs


def build_inverted_index(docs):
    """
    Build a token -> [doc_id, ...] inverted index and an id -> doc lookup dict.
    We index at the whole-document level — each book is one document.
    Adapted from the lab notebook.
    """
    inverted_index = defaultdict(list)
    document_corpus = {}

    for doc in docs:
        document_corpus[doc["id"]] = doc
        # use a set so each token is only added once per document
        tokens = set(simple_tokenize(doc["text"]))
        for token in tokens:
            inverted_index[token].append(doc["id"])

    print(f"Index built with {len(inverted_index)} unique terms.")
    return dict(inverted_index), document_corpus


def save_index(inverted_index, document_corpus, path=INDEX_PICKLE):
    """Save the inverted index and document corpus together as a pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"inverted_index": inverted_index, "document_corpus": document_corpus}, f)
    print(f"Index saved to {path}")


def load_index(path=INDEX_PICKLE):
    """Load the inverted index and document corpus from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Index loaded from {path}")
    return data["inverted_index"], data["document_corpus"]


def build_sqlite_db(books, path=BOOKS_DB):
    """
    Store book metadata in a SQLite database.

    We only store display fields here (title, authors, year, genres) — the
    description and search text live in memory since they're only needed at
    search time. Authors and genres are stored as pipe-separated strings
    because SQLite has no native list type, they're split back on retrieval.
    Potential duplicate IDs are skipped as a safety measure (though this should 
    not be an issue with the current dataset since we use the row index as ID).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()

    # drop and recreate so re-running index.py always gives a clean database
    cur.execute("DROP TABLE IF EXISTS books")
    cur.execute("""
        CREATE TABLE books (
            id      TEXT PRIMARY KEY,
            title   TEXT NOT NULL,
            authors TEXT,
            year    TEXT,
            genres  TEXT
        )
    """)

    # deduplicate by ID before inserting
    seen = set()
    rows = []
    for b in books:
        if b["id"] not in seen:
            seen.add(b["id"])
            rows.append((
                b["id"],
                b["title"],
                "|".join(b["authors"]),
                b["year"],
                "|".join(b["genres"]),
            ))

    cur.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    print(f"SQLite database saved to {path} ({len(rows)} books)")


def lookup_metadata(doc_id, path=BOOKS_DB):
    """Look up display metadata for a single book by ID."""
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT id, title, authors, year, genres FROM books WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    con.close()

    if row is None:
        return None
    return {
        "id":      row[0],
        "title":   row[1],
        "authors": row[2].split("|") if row[2] else [],
        "year":    row[3],
        "genres":  row[4].split("|") if row[4] else [],
    }


if __name__ == "__main__":
    from rank_bm25 import BM25Okapi

    books = load_books()
    docs = build_corpus(books)
    inverted_index, document_corpus = build_inverted_index(docs)

    # save both the pickle index and the SQLite metadata database
    save_index(inverted_index, document_corpus)
    build_sqlite_db(books)

    # quick BM25 sanity check
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    results = bm25.get_top_n(simple_tokenize("dystopian society future"), docs, n=3)
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['title'][:80]}")
