import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from rank_bm25 import BM25Okapi
from preprocess import simple_tokenize


def load_books(path="data/goodreads_books.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# title is repeated twice to give it more weight than the description
def build_corpus(books):
    docs = []
    for b in books:
        docs.append({
            "id":          b["id"],
            "title":       b["title"],
            "description": b["description"],
            "authors":     b["authors"],
            "year":        b["year"],
            "genres":      b["genres"],
            "text": (      b["title"] + " " + 
                            " ".join(b["authors"]) + " " + 
                            " ".join(b["genres"]) + " " +  
                            " ".join(b["genres"]) + " " +
                             b["description"]
)
        })
    return docs


# adapted from the lab notebook
def build_inverted_index(docs):
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
    books = load_books()
    docs = build_corpus(books)
    inverted_index, document_corpus = build_inverted_index(docs)

    # quick check
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    results = bm25.get_top_n(simple_tokenize("dystopian society future"), docs, n=3)
    for i, res in enumerate(results):
        print(f"Rank {i+1}: {res['title'][:80]}...")
