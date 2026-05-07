**ir-books - Goodreads Book Search Engine**


A small information retrieval project built around the Goodreads Best Books Ever dataset. It supports three search methods — Boolean, BM25, and semantic search — and evaluates them using nDCG.


Project structure
```
ir-books/
├── data/                   # created automatically on first run
│   ├── goodreads_books.json
│   └── doc_embeddings.npy
├── src/
│   ├── config.py           # file paths
│   ├── fetch.py            # downloads dataset and saves books to JSON
│   ├── preprocess.py       # tokenizer
│   ├── index.py            # inverted index and BM25 corpus
│   ├── search.py           # search engine (Boolean, BM25, semantic)
│   └── evaluate.py         # nDCG evaluation
├── pyproject.toml
└── README.md
```
Setup
This project uses uv. To install dependencies:
```bash
uv sync
```
Then activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate
```
All commands below should be run from the `src/` directory:
```bash
cd src
```
Running the project
Step 1 — fetch the dataset (only needed once):
```bash
python fetch.py
```
Downloads the Kaggle dataset via `kagglehub` and saves the cleaned books to `data/goodreads_books.json`.
Step 2 — run the search engine:
```bash
python search.py
```
Builds the index, loads the sentence transformer model, and runs ten sample queries through BM25 and semantic search. Document embeddings are cached to `data/doc_embeddings.npy` after the first run, so subsequent runs are faster.
Step 3 — evaluate (optional):
```bash
python evaluate.py
```
Retrieves the top-5 results per query for BM25 and semantic search and prints them with grading prompts. Fill in the relevance scores (0/1/2) in the `MANUAL_SCORES` dict in `evaluate.py`, then run again to compute nDCG@5.
Search methods
Boolean — AND-search over an inverted index; returns all documents containing every query token.
BM25 — ranking based on term frequency and document length normalization.
Semantic — dense retrieval using `all-MiniLM-L6-v2` from `sentence-transformers`, with cosine similarity over pre-computed document embeddings.
