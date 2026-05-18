import os
import re
import ast
import json
import pandas as pd
import kagglehub

from config import BOOKS_JSON


def load_books(max_results=500):
    # download the dataset from Kaggle via kagglehub
    dataset_path = kagglehub.dataset_download("arnabchaki/goodreads-best-books-ever")
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    csv_path = os.path.join(dataset_path, csv_files[0])

    # read all columns as strings to prevent further mangling during parsing
    df = pd.read_csv(csv_path, on_bad_lines="skip", dtype=str)
    print(f"Loaded {len(df)} rows.")

    # normalize column names to lowercase and strip whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    # rename dataset-specific column names to our standard field names
    rename = {
        "author": "authors",
        "desc": "description", "synopsis": "description",
        "genre": "genres",
        "publishdate": "year", "publish date": "year",
        "firstpublishdate": "year", "published": "year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # always use the row index as the document ID — the dataset's ISBN column is unreliable:
    # pandas reads ISBNs as floats, mangling them into scientific notation (e.g. '9.78074E+12'),
    # and about 8% of entries collapse to '1E+13' due to float precision loss, making them
    # non-unique despite the dataset claiming otherwise. a plain integer index is simpler and safe.
    df["id"] = df.index.astype(str)

    # fill in missing columns with empty strings so the rest of the code can assume they exist
    if "authors" not in df.columns: df["authors"] = ""
    if "year"    not in df.columns: df["year"]    = ""
    if "genres"  not in df.columns: df["genres"]  = ""

    # drop books with no title or empty description, then cap the result
    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip().astype(bool)]
    df = df.head(max_results)

    def parse_list(val):
        """Parse a stringified Python list or a comma-separated string into a plain list."""
        val = str(val)
        if val.startswith("["):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
        return [x.strip() for x in val.split(",") if x.strip()]

    books = []
    for _, row in df.iterrows():
        # extract a 4-digit year from whatever date format the dataset uses
        raw_year = str(row.get("year", ""))
        match = re.search(r"\b(1\d{3}|20\d{2})\b", raw_year)
        year = match.group(1) if match else ""

        books.append({
            "id":          str(row["id"]),
            "title":       str(row["title"]).strip(),
            "description": str(row["description"]).strip(),
            "authors":     parse_list(row.get("authors", "")),
            "year":        year,
            "genres":      parse_list(row.get("genres", "")),
        })

    print(f"Done! Got {len(books)} books.")
    return books


if __name__ == "__main__":
    books = load_books(max_results=2000)
    BOOKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(BOOKS_JSON, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print(f"Saved to {BOOKS_JSON}")
