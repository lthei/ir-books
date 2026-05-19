# single source of truth for all queries and their ground-truth book IDs
# imported by search.py, evaluate.py, and lookup_ids.py so queries only
# need to be defined and updated in one place

# ground-truth IDs are filled in by running lookup_ids.py after fetch.py and index.py
# each query maps to a list of two highly relevant book IDs (relevance score = 2)
GROUND_TRUTH = {
    "dystopian society future":      ["38", "20"],  # Brave New World, Fahrenheit 451
    "romance love forbidden":        ["23", "1635"],  # Romeo and Juliet, Twilight
    "mystery detective murder":      ["203", "78"],  # Murder on the Orient Express, The Girl with the Dragon Tattoo
    "fantasy magic dragon":          ["113", "1679"],  # Eragon, A Song of Ice and Fire
    "coming of age young adult":     ["26", "10"],  # The Perks of Being a Wallflower, The Fault in Our Stars
    "historical fiction war":        ["5", "442"],  # The Book Thief, All the Light We Cannot See
    "science fiction aliens":        ["11", "516"],  # The Hitchhiker's Guide to the Galaxy, The War of the Worlds
    "horror supernatural thriller":  ["111", "491"],  # The Shining, The Exorcist
    "biography memoir personal":     ["266", "1406"],  # The Diary of a Young Girl, Long Walk to Freedom
    "philosophy meaning life":       ["24", "239"],  # The Alchemist, Man's Search for Meaning
}

# manual relevance scores for the top-5 results of each query and method
# 2 = highly relevant, 1 = somewhat relevant, 0 = not relevant
# ground-truth books are auto-scored as 2 in evaluate.py, so you only need to judge the rest
# run evaluate.py once first to see the retrieved titles, then fill these in
MANUAL_SCORES = {
    "dystopian society future":      {"bm25": [2, 2, 1, 0, 1],  "semantic": [1, 2, 0, 1, 1]},
    "romance love forbidden":        {"bm25": [1, 0, 0, 1, 1],  "semantic": [1, 0, 1, 1, 1]},
    "mystery detective murder":      {"bm25": [2, 1, 2, 1, 1],  "semantic": [1, 2, 1, 2, 1]},
    "fantasy magic dragon":          {"bm25": [2, 2, 2, 1, 2],  "semantic": [2, 2, 2, 2, 2]},
    "coming of age young adult":     {"bm25": [1, 1, 1, 1, 0],  "semantic": [1, 1, 1, 0, 2]},
    "historical fiction war":        {"bm25": [2, 2, 1, 1, 0],  "semantic": [2, 2, 0, 1, 1]},
    "science fiction aliens":        {"bm25": [2, 2, 1, 1, 1],  "semantic": [1, 1, 2, 1, 2]},
    "horror supernatural thriller":  {"bm25": [2, 1, 2, 2, 1],  "semantic": [2, 2, 0, 2, 0]},
    "biography memoir personal":     {"bm25": [1, 1, 1, 0, 1],  "semantic": [1, 1, 1, 1, 0]},
    "philosophy meaning life":       {"bm25": [2, 2, 1, 0, 1],  "semantic": [1, 1, 2, 2, 1]},
}

# the query strings on their own, for use in search.py and elsewhere
QUERIES = list(GROUND_TRUTH.keys())

# ground-truth book titles, used by lookup_ids.py to find the correct IDs
# keys match GROUND_TRUTH so they can be zipped together
GROUND_TRUTH_TITLES = {
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
