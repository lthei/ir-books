import pandas as pd
import json
import os
import ast
import re
import kagglehub

from config import BOOKS_JSON


def load_books(max_results=500):
    dataset_path = kagglehub.dataset_download("arnabchaki/goodreads-best-books-ever")
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    csv_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"Loaded {len(df)} rows.")

    df.columns = [c.strip().lower() for c in df.columns]

    rename = {
        "bookid": "id", "book id": "id", "isbn": "id",
        "author": "authors",
        "desc": "description", "synopsis": "description",
        "genre": "genres",
        "publishdate": "year", "publish date": "year",
        "firstpublishdate": "year", "published": "year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "id"      not in df.columns: df["id"]      = df.index.astype(str)
    if "authors" not in df.columns: df["authors"] = ""
    if "year"    not in df.columns: df["year"]    = ""
    if "genres"  not in df.columns: df["genres"]  = ""

    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip().astype(bool)]
    df = df.head(max_results)

    def parse_list(val):
        val = str(val)
        if val.startswith("["):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
        return [x.strip() for x in val.split(",") if x.strip()]

    books = []
    for _, row in df.iterrows():
        # Extract 4-digit year from any date format e.g. "10/16/2003" or "October 16th 2003"
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
