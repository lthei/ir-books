# ir-books — Goodreads Book Search Engine


A small information retrieval project built around the Goodreads Best Books Ever dataset. It supports three search methods — Boolean, BM25, and semantic search — and evaluates them using nDCG, Precision, and Recall.


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
│   ├── queries.py              # all queries and ground-truth IDs (single source of truth)
│   ├── fetch.py                # downloads dataset and saves books to JSON
│   ├── preprocess.py           # tokenizer (lowercase, stopwords, etc.)
│   ├── index.py                # index-building functions (used as a module, not run directly)
│   ├── setup.py                # builds all indexes — run once after fetch.py
│   ├── search.py               # search engine (Boolean, BM25, semantic)
│   ├── lookup_ids.py           # helper to find ground-truth book IDs
│   └── evaluate.py             # nDCG, Precision, and Recall evaluation
├── pyproject.toml
└── README.md
```
## Setup
This project uses uv. To install dependencies:
```bash
uv sync
```
Then activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```
All commands below should be run from the `src/` directory:
```bash
cd src
```
## Running the project
**Step 1 — fetch the dataset (only needed once, or to refresh the data):**

```bash
python fetch.py
```
Downloads the Kaggle dataset via `kagglehub` and saves the cleaned books to `data/goodreads_books.json`. Row indices are used as document IDs instead of ISBNs, which are unreliable in this dataset (pandas reads them as floats, and ~8% collapse to the same value).

**Step 2 — build all indexes:**

```bash
python setup.py
```
Builds the inverted index (saved as `data/inverted_index.pkl`), the SQLite metadata database (`data/books.db`), and the sentence-transformer embeddings cache (`data/doc_embeddings.npy`). Run this after every `fetch.py`. The embeddings step takes a few minutes on first run.

**Step 3 — run the search engine:**

```bash
python search.py
```
Loads all indexes and runs the queries defined in `queries.py` through BM25 and semantic search.

**Step 4 — find ground-truth IDs (only needed once, or after re-fetching):**

```bash
python lookup_ids.py
```
Looks up the dataset IDs for all ground-truth books defined in `queries.py` and prints a ready-to-paste `GROUND_TRUTH` dict. Copy the output into `GROUND_TRUTH` in `queries.py`. For any books marked `NOT_FOUND`, search manually in `goodreads_books.json` or with:
```bash
python -c "from lookup_ids import find_book; print(find_book('title fragment'))"
```
**Step 5 — evaluate:**

```bash
python evaluate.py
```
Prints the top-5 results per query with grading prompts. Fill in the relevance scores (0/1/2) in `MANUAL_SCORES` in `evaluate.py`, then run again to compute nDCG@5, Precision@5, and Recall@5. Ground-truth books are automatically scored as 2.
Adding or changing queries
All queries and ground-truth data live in `queries.py`. To add or change a query:
Update `GROUND_TRUTH`, `QUERIES`, and `GROUND_TRUTH_TITLES` in `queries.py`
Run `lookup_ids.py` to find IDs for any new ground-truth books
Add a matching entry to `MANUAL_SCORES` in `evaluate.py`
No other files need to be changed.
## Indexing
We index at the whole-document level — each book is one document. The search text field concatenates title, authors, genres (repeated twice for extra weight), and description. Year is stored for display only and excluded from search to avoid date-based ranking bias.
The index is persisted in two forms:
Pickle (`inverted_index.pkl`) — stores the token → document ID mapping and the full document corpus for fast in-memory lookup
SQLite (`books.db`) — stores structured metadata for display; makes it easy to look up book information by ID without loading the full corpus
## Search methods
- Boolean — AND-search over the inverted index; returns all documents containing every query token
- BM25 — ranking based on term frequency and document length normalization
- Semantic — dense retrieval using `all-MiniLM-L6-v2` from `sentence-transformers`, with cosine similarity over pre-computed document embeddings
## Evaluation
Results are evaluated using three metrics, all computed at rank 5:
- nDCG@5 — measures whether the better results are ranked higher
- Precision@5 — fraction of the top-5 results that are ground-truth books
- Recall@5 — fraction of ground-truth books that appear in the top-5 results
