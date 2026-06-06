import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

from preprocess import simple_tokenize, expand_query
from index import load_books, build_corpus, load_index, lookup_metadata
from config import EMBEDDINGS_NPY, INDEX_PICKLE, BOOKS_DB
from queries import QUERY_STRINGS

class BookSearchEngine:

    """
    Wraps all four search methods (Boolean, BM25, semantic, RRF) in a class
    because they all share the same expensive setup: loading the corpus,
    building the index, loading the model, and encoding the documents.
    Using a class means we do that setup once in __init__ and reuse it
    across all methods, rather than rebuilding everything each time
    or relying on global variables.
    """

    def __init__(self):
        # load books and build the document corpus
        books = load_books()
        self.docs = build_corpus(books)

        # load the inverted index — setup.py must be run first to create it
        if not INDEX_PICKLE.exists():
            raise FileNotFoundError(
                f"Index not found at {INDEX_PICKLE}. Run setup.py first."
            )
        self.inverted_index, self.document_corpus = load_index()

        # build the BM25 index from the tokenized corpus
        tokenized_corpus = [simple_tokenize(doc["text"]) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # load the sentence transformer model for semantic search
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # load the pre-computed document embeddings — setup.py must be run first to create them
        if not EMBEDDINGS_NPY.exists():
            raise FileNotFoundError(
                f"Embeddings not found at {EMBEDDINGS_NPY}. Run setup.py first."
            )
        print("Loading embeddings from cache...")
        self.doc_embeddings = np.load(EMBEDDINGS_NPY)

    def boolean_search(self, query, expand=False): # adapted from colab notebook
        """AND-Boolean search over the inverted index."""
        if expand:
            query = expand_query(query)

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        # start with all docs containing the first token, then intersect with the rest
        results = set(self.inverted_index.get(query_tokens[0], []))
        for token in query_tokens[1:]:
            results &= set(self.inverted_index.get(token, []))
        return [self.document_corpus[doc_id] for doc_id in results]

    def bm25_search(self, query, n=5, expand=False): # adapted from colab notebook
        """Return the top-n BM25-ranked results for the given query."""
        if expand:
            query = expand_query(query)

        tokenized_query = simple_tokenize(query)
        if not tokenized_query:
            return []
        return self.bm25.get_top_n(tokenized_query, self.docs, n=n)

    def semantic_search(self, query, top_k=5, expand=False): # adapted from colab notebook
        """Return the top-k semantically similar results using cosine similarity."""
        if expand:
            query = expand_query(query)

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]
        top_indices = scores.topk(k=top_k).indices
        return [self.docs[i] for i in top_indices]

    def rrf_search(self, query, n=5, k=60, expand=False):
        """Fuse BM25 and semantic rankings using Reciprocal Rank Fusion (RRF).

        RRF combines two ranked lists into a single ranking without needing
        to normalise or calibrate their scores against each other. Each
        document receives a score of 1 / (k + rank) from each list it appears
        in, and the scores are summed. Documents that rank highly in both
        lists end up with a higher combined score than documents that only
        do well in one.

        The smoothing constant k (default 60, from Cormack et al. 2009)
        controls how much weight high ranks get relative to lower ones.
        A larger k flattens the curve; a smaller k amplifies the top ranks.
        """
        if expand:
            query = expand_query(query)

        # retrieve a larger candidate pool than the final n so the fusion
        # has enough material to re-rank — using n * 3 is a common heuristic
        pool = max(n * 3, 20)

        # expansion is already applied above, so pass the (possibly expanded)
        # query string directly and leave expand=False to avoid expanding twice
        bm25_results = self.bm25_search(query,    n=pool)
        sem_results  = self.semantic_search(query, top_k=pool)

        # build a {doc_id: rank} lookup for each method (ranks are 1-based
        # so the lowest rank value = highest relevance, matching the 1/(k+r) formula)
        bm25_ranks = {doc["id"]: rank for rank, doc in enumerate(bm25_results, start=1)}
        sem_ranks  = {doc["id"]: rank for rank, doc in enumerate(sem_results,  start=1)}

        # collect all unique doc IDs seen by either ranker
        all_ids = set(bm25_ranks) | set(sem_ranks)

        # compute the RRF score for each candidate document:
        # documents missing from one list contribute 0 from that list
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc_id])
            if doc_id in sem_ranks:
                score += 1.0 / (k + sem_ranks[doc_id])
            rrf_scores[doc_id] = score

        # sort by descending RRF score and return the top-n doc dicts
        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:n]
        return [self.document_corpus[doc_id] for doc_id in top_ids
                if doc_id in self.document_corpus]


def _format_result(rank, doc):
    # fetch display metadata from SQLite for a cleaner separation between search and display
    meta = lookup_metadata(doc["id"]) or doc
    authors = ", ".join(meta["authors"]) if meta["authors"] else "Unknown"
    year = f" ({meta['year']})" if meta["year"] else ""
    print(f"  Rank {rank}: {meta['title']} by {authors}{year}")
    print(f"    {doc['description'][:100]}...")

if __name__ == "__main__":
    engine = BookSearchEngine()

    # queries are imported from queries.py so they only need to be defined in one place
    for query in QUERY_STRINGS:
        print(f"\n{'='*60}")
        print(f"> {query}")

        print("\n  BM25 Search:")
        for i, res in enumerate(engine.bm25_search(query, n=5), start=1):
            _format_result(i, res)

        print("\n  Semantic Search:")
        for i, res in enumerate(engine.semantic_search(query, top_k=5), start=1):
            _format_result(i, res)

        print("\n  RRF Search:")
        for i, res in enumerate(engine.rrf_search(query, n=5), start=1):
            _format_result(i, res)

        bool_hits = engine.boolean_search(query)
        if bool_hits:
            print(f"\n  Boolean hits: {len(bool_hits)}")
        else:
            print("\n  Boolean hits: 0 (no document contains all query terms)")
