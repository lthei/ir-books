import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from preprocess import simple_tokenize
from index import load_books, build_corpus, build_inverted_index
from config import EMBEDDINGS_NPY


class BookSearchEngine:
    """
    Wraps all three search methods (Boolean, BM25, semantic) in a class
    because they all share the same expensive setup: loading the corpus,
    building the index, loading the model, and encoding the documents.
    Using a class means we do that setup once in __init__ and reuse it
    across all three methods, rather than rebuilding everything each time
    or relying on global variables.
    """

    def __init__(self):
        # load and index books
        books = load_books()
        self.docs = build_corpus(books)
        self.inverted_index, self.document_corpus = build_inverted_index(self.docs)

        # BM25 index
        tokenized_corpus = [simple_tokenize(doc["text"]) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # load the sentence transformer model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # encode all documents — load from cache if available to save time
        if EMBEDDINGS_NPY.exists():
            print("Loading embeddings from cache...")
            self.doc_embeddings = np.load(EMBEDDINGS_NPY)
        else:
            print("Encoding documents (this may take a while)...")
            doc_texts = [doc["text"] for doc in self.docs]
            self.doc_embeddings = self.model.encode(
                doc_texts, convert_to_numpy=True, show_progress_bar=True
            )
            EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_NPY, self.doc_embeddings)
            print(f"Saved embeddings to {EMBEDDINGS_NPY}")

    def boolean_search(self, query):
        """AND-Boolean search over the inverted index."""
        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        results = set(self.inverted_index.get(query_tokens[0], []))
        for token in query_tokens[1:]:
            results &= set(self.inverted_index.get(token, []))

        return [self.document_corpus[doc_id] for doc_id in results]

    def bm25_search(self, query, n=5):
        """Return the top-n BM25-ranked results for the given query."""
        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
        return self.bm25.get_top_n(tokenized_query, self.docs, n=n)

    def semantic_search(self, query, top_k=5):
        """Return the top-k semantically similar results for the given query."""
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        top_indices = scores.topk(k=top_k).indices
        return [self.docs[i] for i in top_indices]


def _format_result(rank, doc):
    authors = ", ".join(doc["authors"]) if doc["authors"] else "Unknown"
    year = f" ({doc['year']})" if doc["year"] else ""
    print(f"  Rank {rank}: {doc['title']} by {authors}{year}")
    print(f"           {doc['description'][:100]}...")


if __name__ == "__main__":
    engine = BookSearchEngine()

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
        print(f"\n{'='*60}")
        print(f"> {query}")

        print("\n  BM25 Search:")
        for i, res in enumerate(engine.bm25_search(query, n=5), start=1):
            _format_result(i, res)

        print("\n  Semantic Search:")
        for i, res in enumerate(engine.semantic_search(query, top_k=5), start=1):
            _format_result(i, res)

        bool_hits = engine.boolean_search(query)
        if bool_hits:
            print(f"\n  Boolean hits: {len(bool_hits)}")
        else:
            print("\n  Boolean hits: 0 (no document contains all query terms)")