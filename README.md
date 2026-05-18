# ir-books — Goodreads Book Search Engine

A small information retrieval project built around the [Goodreads Best Books Ever](https://www.kaggle.com/datasets/arnabchaki/goodreads-best-books-ever) dataset. It supports three search methods — Boolean, BM25, and semantic search — and evaluates them using nDCG, Precision, and Recall.

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
│   ├── fetch.py                # downloads dataset and saves books to JSON
│   ├── preprocess.py           # tokenizer (lowercase, stopwords, etc.)
│   ├── index.py                # inverted index, pickle, and SQLite database
│   ├── search.py               # search engine (Boolean, BM25, semantic)
│   ├── lookup_ids.py           # helper to find ground-truth book IDs
│   └── evaluate.py             # nDCG, Precision, and Recall evaluation
├── pyproject.toml
└── README.md
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv). To install dependencies:

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

**Step 1 — fetch the dataset** (only needed once, or to refresh the data):
```bash
python fetch.py
```
Downloads the Kaggle dataset via `kagglehub` and saves the cleaned books to `data/goodreads_books.json`. All columns are read as strings to prevent pandas from mangling ISBNs as floats.

**Step 2 — build the index:**
```bash
python index.py
```
Builds the inverted index and saves it as `data/inverted_index.pkl`. Also creates the SQLite metadata database at `data/books.db`, which stores title, authors, year, and genres per book. Run this after every `fetch.py`.

**Step 3 — run the search engine:**
```bash
python search.py
```
Loads the index and runs ten sample queries through BM25 and semantic search. Document embeddings are cached to `data/doc_embeddings.npy` after the first run so subsequent runs are faster.

**Step 4 — evaluate (optional):**

First, look up the correct IDs for the ground-truth books:
```bash
python lookup_ids.py
```
Copy the printed `GROUND_TRUTH` dict into `evaluate.py`. For any books marked `NOT_FOUND`, look them up manually in `books.db` using a SQLite browser (e.g. [DB Browser for SQLite](https://sqlitebrowser.org)) or with:
```bash
python -c "from index import find_ids_by_title; print(find_ids_by_title('fragment'))"
```

Then run the evaluation:
```bash
python evaluate.py
```
Prints the top-5 results per query with grading prompts. Fill in the relevance scores (0/1/2) in `MANUAL_SCORES` in `evaluate.py`, then run again to compute nDCG@5, Precision@5, and Recall@5. Ground-truth books are automatically scored as 2.

## Indexing

We index at the **whole-document level** — each book is one document. The search text field concatenates title, authors, genres (weighted by repetition), and description. Year is stored for display only and excluded from search to avoid date-based ranking bias.

The index is persisted in two forms:
- **Pickle** (`inverted_index.pkl`) — stores the token → document ID mapping and the full document corpus for fast in-memory lookup
- **SQLite** (`books.db`) — stores structured metadata for display; makes it easy to look up book information by ID without loading the full corpus

## Search methods

- **Boolean** — AND-search over the inverted index; returns all documents containing every query token
- **BM25** — ranking based on term frequency and document length normalization
- **Semantic** — dense retrieval using `all-MiniLM-L6-v2` from `sentence-transformers`, with cosine similarity over pre-computed document embeddings

## Evaluation

Results are evaluated using three metrics, all computed at rank 5:
- **nDCG@5** — measures whether the better results are ranked higher
- **Precision@5** — fraction of the top-5 results that are ground-truth books
- **Recall@5** — fraction of ground-truth books that appear in the top-5 results
