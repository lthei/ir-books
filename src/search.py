import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from rank_bm25 import BM25Okapi
from preprocess import simple_tokenize
from index import load_books, build_corpus, build_inverted_index


# from the lab notebook
def boolean_search(query, index):
    query_tokens = simple_tokenize(query)
    if not query_tokens:
        return []

    results = set(index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        results = results.intersection(set(index.get(token, [])))

    return list(results)


# adapted from the lab notebook
def bm25_search(query, n=5):
    tokenized_query = simple_tokenize(query)
    top_n = bm25.get_top_n(tokenized_query, docs, n=n)
    return top_n


# --- setup ---
books = load_books()
docs = build_corpus(books)
inverted_index, document_corpus = build_inverted_index(docs)

tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

# --- queries ---
queries = [
    "dystopian society future",
    "romance love forbidden",
    "mystery detective murder",
    "fantasy magic dragon",
    "coming of age young adult",
    "historical fiction war",
    "science fiction aliens",
    "horror supernatural thriller",
    "biography memoir personal",
    "philosophy meaning life",
]

for query in queries:
    results = bm25_search(query)
    print(f"\nQuery: '{query}'")
    for i, res in enumerate(results):
        authors = ", ".join(res["authors"]) if res["authors"] else "Unknown"
        print(f"  Rank {i+1}: {res['title']} by {authors} ({res['year']})")
        print(f"           {res['description'][:100]}...")

    # also show boolean hit count for comparison
    bool_hits = boolean_search(query, inverted_index)
    print(f"  Boolean hits: {len(bool_hits)}")
