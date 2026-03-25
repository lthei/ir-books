import pandas as pd
import json
import os
import ast
import kagglehub


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
    if "rating"  not in df.columns: df["rating"]  = ""

    df = df.dropna(subset=["title", "description"])
    df = df[df["description"].str.strip().astype(bool)]
    df = df.head(max_results)

    books = []
    for _, row in df.iterrows():
        # extract 4-digit year from strings like "October 16th 2003"
        year = ""
        for part in str(row.get("year", "")).split():
            if part.isdigit() and len(part) == 4:
                year = part
                break

        # genres and authors may be stringified Python lists
        def parse_list(val):
            val = str(val)
            if val.startswith("["):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return [val]
            return [x.strip() for x in val.split(",") if x.strip()]

        books.append({
            "id":          str(row["id"]),
            "title":       str(row["title"]).strip(),
            "description": str(row["description"]).strip(),
            "authors":     parse_list(row.get("authors", "")),
            "year":        year,
            "genres":      parse_list(row.get("genres", "")),
            "rating":      str(row.get("rating", "")).strip(),
        })

    print(f"Done! Got {len(books)} books.")
    return books


if __name__ == "__main__":
    books = load_books(max_results=500)
    with open("data/goodreads_books.json", "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)
    print("Saved to data/goodreads_books.json")
