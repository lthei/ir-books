import json
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from preprocess import simple_tokenize
from index import load_books, build_corpus, build_inverted_index


def boolean_search(query, index):
    query_tokens = simple_tokenize(query)
    if not query_tokens:
        return []

    results = set(index.get(query_tokens[0], []))
    for token in query_tokens[1:]:
        results = results.intersection(set(index.get(token, [])))

    return list(results)


def bm25_search(query, n=5):
    tokenized_query = simple_tokenize(query)
    return bm25.get_top_n(tokenized_query, docs, n=n)


def semantic_search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_results = scores.topk(k=top_k).indices
    return [docs[i] for i in top_results]


# setup
books = load_books()
docs = build_corpus(books)
inverted_index, document_corpus = build_inverted_index(docs)

# BM25 index
tokenized_corpus = [simple_tokenize(doc["text"]) for doc in docs]
bm25 = BM25Okapi(tokenized_corpus)

# model
model = SentenceTransformer("all-MiniLM-L6-v2")

# document encoding — load from cache if available, otherwise encode and save
EMBEDDINGS_CACHE = "data/doc_embeddings.npy"

if os.path.exists(EMBEDDINGS_CACHE):
    print("Loading embeddings from cache...")
    doc_embeddings = np.load(EMBEDDINGS_CACHE)
else:
    print("Encoding documents (this may take a while)...")
    doc_texts = [doc["text"] for doc in docs]
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMBEDDINGS_CACHE, doc_embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_CACHE}")

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
    print(f"\n> {query}")

    print("\nBM25 Search:")
    bm25_results = bm25_search(query, n=5)
    for i, res in enumerate(bm25_results):
        authors = ", ".join(res["authors"]) if res["authors"] else "Unknown"
        print(f"  Rank {i+1}: {res['title']} by {authors} ({res['year']})")
        print(f"           {res['description'][:100]}...")

    print("\nSemantic Search:")
    semantic_results = semantic_search(query, top_k=5)
    for i, res in enumerate(semantic_results):
        authors = ", ".join(res["authors"]) if res["authors"] else "Unknown"
        print(f"  Rank {i+1}: {res['title']} by {authors} ({res['year']})")
        print(f"           {res['description'][:100]}...")

    bool_hits = boolean_search(query, inverted_index)
    print(f"\n  Boolean hits: {len(bool_hits)}")