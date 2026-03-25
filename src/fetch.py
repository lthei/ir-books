import pandas as pd
import json
import os
import kagglehub


def load_books(max_results=500):
    """
    Load books from the Goodreads Best Books Ever dataset via kagglehub.

    On first run this downloads and caches the dataset automatically.
    Requires a Kaggle account — kagglehub will prompt for credentials if needed.

    Dataset: https://www.kaggle.com/datasets/arnabchaki/goodreads-best-books-ever
    Columns: bookId, title, author, rating, description, genres,
             pages, publishDate, firstPublishDate, ...
    """
    print("Fetching dataset via kagglehub (cached after first run)...")
    dataset_path = kagglehub.dataset_download("arnabchaki/goodreads-best-books-ever")
    print(f"Dataset path: {dataset_path}")

    # Find the CSV file in the downloaded directory
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_path}")
    csv_path = os.path.join(dataset_path, csv_files[0])
    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"Loaded {len(df)} rows from {csv_path}")

    # --- normalise column names to lowercase/stripped ---
    df.columns = [c.strip().lower() for c in df.columns]

    # Map common column name variants to our standard names
    rename = {
        "bookid":           "id",
        "book id":          "id",
        "isbn":             "id",
        "author":           "authors",
        "description":      "description",
        "desc":             "description",
        "synopsis":         "description",
        "genre":            "genres",
        "publishdate":      "year",
        "publish date":     "year",
        "firstpublishdate": "year",
        "published":        "year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure required columns exist
    for col in ("title", "description"):
        if col not in df.columns:
            raise ValueError(f"Could not find a '{col}' column. Available: {list(df.columns)}")

    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    if "authors" not in df.columns:
        df["authors"] = ""

    if "year" not in df.columns:
        df["year"] = ""

    if "genres" not in df.columns:
        df["genres"] = ""

    if "rating" not in df.columns:
        df["rating"] = ""

    # Drop rows without a description
    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip().astype(bool)]

    # Limit to max_results
    df = df.head(max_results)

    books = []
    for _, row in df.iterrows():
        # Parse year from date strings like "October 16th 2003" or "2003"
        raw_year = str(row.get("year", ""))
        year = ""
        for part in raw_year.split():
            if part.isdigit() and len(part) == 4:
                year = part
                break

        # Genres: may be a stringified list or comma-separated
        raw_genres = str(row.get("genres", ""))
        if raw_genres.startswith("["):
            try:
                import ast
                genres = ast.literal_eval(raw_genres)
            except Exception:
                genres = [raw_genres]
        else:
            genres = [g.strip() for g in raw_genres.split(",") if g.strip()]

        # Authors: same treatment
        raw_authors = str(row.get("authors", ""))
        if raw_authors.startswith("["):
            try:
                import ast
                authors = ast.literal_eval(raw_authors)
            except Exception:
                authors = [raw_authors]
        else:
            authors = [a.strip() for a in raw_authors.split(",") if a.strip()]

        books.append({
            "id":          str(row["id"]),
            "title":       str(row["title"]).strip(),
            "description": str(row["description"]).strip(),
            "authors":     authors,
            "year":        year,
            "genres":      genres,
            "rating":      str(row.get("rating", "")).strip(),
        })

    print(f"Done! Prepared {len(books)} books.")
    return books


if __name__ == "__main__":
    books = load_books(max_results=500)
    os.makedirs("data", exist_ok=True)
    with open("data/goodreads_books.json", "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print("Saved to data/goodreads_books.json")
