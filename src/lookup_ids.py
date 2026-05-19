import json
from config import BOOKS_JSON

# this is a helper script to find the correct IDs for the ground-truth books in evaluate.py;
# run this script after fetch.py and index.py to easily find the correct IDs
# for all ground-truth books, then copy the output into evaluate.py

ground_truth = {
    "dystopian society future":      ["Brave New World", "Fahrenheit 451"],
    "romance love forbidden":        ["Romeo and Juliet", "Twilight"],
    "mystery detective murder":      ["Murder on the Orient Express", "The Girl with the Dragon Tattoo"],
    "fantasy magic dragon":          ["Eragon", "A Song of Ice and Fire"],
    "coming of age young adult":     ["The Perks of Being a Wallflower", "The Fault in Our Stars"],
    "historical fiction war":        ["The Book Thief", "All the Light We Cannot See"],
    "science fiction aliens":        ["The Hitchhiker's Guide to the Galaxy", "The War of the Worlds"],
    "horror supernatural thriller":  ["The Shining", "The Exorcist"],
    "biography memoir personal":     ["The Diary of a Young Girl", "Long Walk to Freedom"],
    "philosophy meaning life":       ["The Alchemist", "Man's Search for Meaning"],
}

# load the books directly from JSON — IDs are exactly as stored, no database involved
with open(BOOKS_JSON, "r", encoding="utf-8") as f:
    books = json.load(f)

# build a lookup: lowercase title -> book, for case-insensitive matching
title_index = {b["title"].lower(): b for b in books}


def find_book(title):
    """Look up a book by exact title (case-insensitive). Returns the book dict or None."""
    return title_index.get(title.lower())


print("GROUND_TRUTH = {")
for query, titles in ground_truth.items():
    ids = []
    for title in titles:
        book = find_book(title)
        if book:
            ids.append(f'"{book["id"]}"')
            print(f"    # found: {book['title']} → {book['id']}")
        else:
            ids.append('"NOT_FOUND"')
            print(f"    # NOT FOUND: {title}")
            print(f"    #   -> search the JSON manually: \"{title}\" data/goodreads_books.json")
    print(f'    "{query}": [{", ".join(ids)}],')
print("}")
