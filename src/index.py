import json
from collections import defaultdict

from preprocess import simple_tokenize
from config import BOOKS_JSON


def load_books(path=BOOKS_JSON):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(books):
    """
    Note: The 'text' field is used for indexing and ranking only — it intentionally
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
    Build a token → [doc_id, ...] inverted index and an id → doc lookup dict.
    Adapted from the lab notebook.
    """
    inverted_index = defaultdict(list)
    document_corpus = {}

    for doc in docs:
        document_corpus[doc["id"]] = doc
        tokens = set(simple_tokenize(doc["text"]))
        for token in tokens:
            inverted_index[token].append(doc["id"])

    print(f"Index built with {len(inverted_index)} unique terms.")
    return dict(inverted_index), document_corpus


if __name__ == "__main__":
    from rank_bm25 import BM25Okapi

    books = load_books()
    docs = build_corpus(books)
    inverted_index, document_corpus = build_inverted_index(docs)

    # BM25 sanity check
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    results = bm25.get_top_n(simple_tokenize("dystopian society future"), docs, n=3)
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['title'][:80]}")
