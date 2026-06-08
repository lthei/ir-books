# ir-books — Goodreads Book Search Engine

A small information retrieval project built around the Goodreads Best Books Ever dataset. It supports four search methods — Boolean, BM25, semantic search, and Reciprocal Rank Fusion (RRF) — and evaluates them using nDCG, Precision, and Recall.

## Project structure

```
ir-books/
├── data/                       # created automatically on first run
│   ├── goodreads_books.json    # cleaned book data
│   ├── inverted_index.pkl      # pickled inverted index + document corpus
│   ├── doc_embeddings.npy      # cached sentence-transformer embeddings
│   └── books.db                # SQLite metadata database
├── src/
│   ├── config.py               # centralized file paths
│   ├── queries.py              # all queries, ground-truth IDs, and manual scores (single source of truth)
│   ├── fetch.py                # downloads dataset and saves books to JSON
│   ├── preprocess.py           # tokenizer (lowercase, stopwords, etc.) and WordNet query expansion
│   ├── index.py                # index-building functions (used as a module, not run directly)
│   ├── setup.py                # builds all indexes — run once after fetch.py
│   ├── search.py               # search engine (Boolean, BM25, semantic, RRF)
│   ├── lookup_ids.py           # helper to find ground-truth book IDs
│   ├── evaluate.py             # nDCG, Precision, and Recall evaluation
│   └── app.py                  # Streamlit GUI
├── pyproject.toml
└── README.md
```

## Setup

This project uses uv. To install dependencies:

```
uv sync
```

Then activate the virtual environment:

```
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

All commands below should be run from the `src/` directory:

```
cd src
```

## Running the project

**Step 1 — fetch the dataset (only needed once, or to refresh the data):**

```
python fetch.py
```

Downloads the Kaggle dataset via `kagglehub` and saves the cleaned books to `data/goodreads_books.json`. Row indices are used as document IDs instead of ISBNs, which are unreliable in this dataset (they are read as floats, and ~8% collapse to the same value).

**Step 2 — build all indexes (run after every fetch):**

```
python setup.py
```

Builds the inverted index (data/inverted_index.pkl), the SQLite metadata database (data/books.db), and the sentence-transformer embeddings cache (data/doc_embeddings.npy). The embeddings step takes a few minutes on first run but is cached for subsequent runs.

**Step 3 — run the search engine:**

```
python search.py
```

Loads all indexes and runs the queries defined in `queries.py` through BM25, semantic, and RRF search.

**Step 4 — find ground-truth IDs (only needed once, or after re-fetching):**

```
python lookup_ids.py
```

Looks up the dataset IDs for all ground-truth books defined in queries.py and prints a ready-to-paste GROUND_TRUTH dict. Copy the gt_ids values into the corresponding entries in queries.py. For any books marked NOT_FOUND, search manually in goodreads_books.json or with:

```
python -c "from lookup_ids import find_book; print(find_book('title fragment'))"
```

**Step 5 — evaluate:**

```
python evaluate.py
```

Prints the top-5 results per query with grading prompts. Fill in the relevance scores (0/1/2) in the manual_scores field of each query entry in queries.py, then run again to compute nDCG@5, Precision@5, and Recall@5 for all three methods. Ground-truth books are automatically scored as 2.

**Step 6 — run the GUI:**

```
streamlit run app.py
```

Opens a browser-based interface for interactive search. The sidebar lets you choose a search method, toggle query expansion, and set the number of results. Results are displayed as cards showing title, author, year, genres, and a description excerpt.

**Adding or changing queries**

All query data lives in the QUERIES list in queries.py. Each entry contains the query string, ground-truth titles, ground-truth IDs, and manual scores. To add a new query:

- Add a new entry to QUERIES in queries.py
- Run lookup_ids.py to find the IDs for the new ground-truth books
- Fill in gt_ids and manual_scores in the new entry

No other files need to be changed.

## Indexing

We index at the whole-document level — each book is one document. The search text field concatenates title, authors, genres (repeated twice for extra weight), and description. Year is stored for display only and excluded from search to avoid date-based ranking bias. Books without a title or description are dropped during preprocessing.
The index is persisted in two forms:
Pickle (`inverted_index.pkl`) — stores the token → document ID mapping and the full document corpus for fast in-memory lookup
SQLite (`books.db`) — stores structured metadata for display; makes it easy to look up book information by ID without loading the full corpus

## Search methods

- Boolean — AND-search over the inverted index; returns all documents containing every query token, capped at the selected number of results
- BM25 — ranking based on term frequency and document length normalization
- Semantic — dense retrieval using `all-MiniLM-L6-v2` from `sentence-transformers`, with cosine similarity over pre-computed document embeddings
- RRF — Reciprocal Rank Fusion; combines the BM25 and Semantic rankings into a single list. Each document receives a score of `1 / (k + rank)` from each list it appears in, and the scores are summed. Documents that rank highly in both methods score highest.

## Query expansion

All search methods support optional WordNet query expansion. When enabled, `expand_query()` from `preprocess.py` appends synonyms for each content word in the query before searching, which can improve recall when relevant books use different but related vocabulary. The original query terms are always preserved.

To activate it, pass `expand=True` to any search method:

```python
engine.bm25_search("mystery detective", expand=True)
engine.semantic_search("mystery detective", expand=True)
engine.rrf_search("mystery detective", expand=True)
engine.boolean_search("mystery detective", expand=True)
```

By default, `expand=False` and queries are passed through unchanged. In the GUI, expansion is toggled via the sidebar.

## Evaluation

Results are evaluated using three metrics, all computed at rank 5:

- nDCG@5 — measures whether the better results are ranked higher
- Precision@5 — fraction of the top-5 results that are ground-truth books
- Recall@5 — fraction of ground-truth books that appear in the top-5 results
