import json
from config import BOOKS_JSON
from queries import GROUND_TRUTH_TITLES

# run this script after fetch.py and setup.py to find the correct IDs
# for all ground-truth books, then copy the printed GROUND_TRUTH dict into queries.py

# load books directly from JSON — IDs are exactly as stored, no database involved
with open(BOOKS_JSON, "r", encoding="utf-8") as f:
    books = json.load(f)

# build a lowercase title → book lookup for case-insensitive exact matching
title_index = {b["title"].lower(): b for b in books}


def find_book(title):
    """Look up a book by exact title (case-insensitive). Returns the book dict or None."""
    return title_index.get(title.lower())


print("# copy this into GROUND_TRUTH in queries.py")
print("GROUND_TRUTH = {")
for query, titles in GROUND_TRUTH_TITLES.items():
    ids = []
    for title in titles:
        book = find_book(title)
        if book:
            ids.append(f'"{book["id"]}"')
            print(f"    # found: {book['title']} → {book['id']}")
        else:
            ids.append('"NOT_FOUND"')
            print(f"    # NOT FOUND: {title}")
            print(f"    #   → search manually: grep -i \"{title}\" ../data/goodreads_books.json")
    print(f'    "{query}": [{", ".join(ids)}],')
print("}")
