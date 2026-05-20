# single source of truth for all evaluation queries
# to add or change a query, edit the QUERIES list below — nothing else needs to change
#
# each entry has:
#   query         — the search string
#   gt_titles     — ground-truth book titles (for lookup_ids.py)
#   gt_ids        — ground-truth book IDs (fill in by running lookup_ids.py)
#   manual_scores — relevance scores (0/1/2) for the top-5 retrieved results per method
#                   run evaluate.py once first to see which titles were retrieved,
#                   then fill in the scores here (ground-truth books are auto-scored as 2)
#
# manual scores: 0 = not relevant, 1 = somewhat relevant, 2 = highly relevant

QUERIES = [
    {
        "query":         "dystopian society future",
        "gt_titles":     ["Brave New World", "Fahrenheit 451"],
        "gt_ids":        ["38", "20"],
        "manual_scores": {"bm25": [2, 2, 1, 0, 1],  "semantic": [1, 2, 0, 1, 1]},
    },
    {
        "query":         "romance love forbidden",
        "gt_titles":     ["Romeo and Juliet", "Twilight"],
        "gt_ids":        ["23", "1635"],
        "manual_scores": {"bm25": [1, 0, 0, 1, 1],  "semantic": [1, 0, 1, 1, 1]},
    },
    {
        "query":         "mystery detective murder",
        "gt_titles":     ["Murder on the Orient Express", "The Girl with the Dragon Tattoo"],
        "gt_ids":        ["203", "78"],
        "manual_scores": {"bm25": [2, 1, 2, 1, 1],  "semantic": [1, 2, 1, 2, 1]},
    },
    {
        "query":         "fantasy magic dragon",
        "gt_titles":     ["Eragon", "A Song of Ice and Fire"],
        "gt_ids":        ["113", "1679"],
        "manual_scores": {"bm25": [2, 2, 2, 1, 2],  "semantic": [2, 2, 2, 2, 2]},
    },
    {
        "query":         "coming of age young adult",
        "gt_titles":     ["The Perks of Being a Wallflower", "The Fault in Our Stars"],
        "gt_ids":        ["26", "10"],
        "manual_scores": {"bm25": [1, 1, 1, 1, 0],  "semantic": [1, 1, 1, 0, 2]},
    },
    {
        "query":         "historical fiction war",
        "gt_titles":     ["The Book Thief", "All the Light We Cannot See"],
        "gt_ids":        ["5", "442"],
        "manual_scores": {"bm25": [2, 2, 1, 1, 0],  "semantic": [2, 2, 0, 1, 1]},
    },
    {
        "query":         "science fiction aliens",
        "gt_titles":     ["The Hitchhiker's Guide to the Galaxy", "The War of the Worlds"],
        "gt_ids":        ["11", "516"],
        "manual_scores": {"bm25": [2, 2, 1, 1, 1],  "semantic": [1, 1, 2, 1, 2]},
    },
    {
        "query":         "horror supernatural thriller",
        "gt_titles":     ["The Shining", "The Exorcist"],
        "gt_ids":        ["111", "491"],
        "manual_scores": {"bm25": [2, 1, 2, 2, 1],  "semantic": [2, 2, 0, 2, 0]},
    },
    {
        "query":         "biography memoir personal",
        "gt_titles":     ["The Diary of a Young Girl", "Long Walk to Freedom"],
        "gt_ids":        ["266", "1406"],
        "manual_scores": {"bm25": [1, 1, 1, 0, 1],  "semantic": [1, 1, 1, 1, 0]},
    },
    {
        "query":         "philosophy meaning life",
        "gt_titles":     ["The Alchemist", "Man's Search for Meaning"],
        "gt_ids":        ["24", "239"],
        "manual_scores": {"bm25": [2, 2, 1, 0, 1],  "semantic": [1, 1, 2, 2, 1]},
    },
]

# convenience structures derived from QUERIES so other files don't need to loop themselves
QUERY_STRINGS  = [q["query"] for q in QUERIES]
GROUND_TRUTH   = {q["query"]: q["gt_ids"]        for q in QUERIES}
MANUAL_SCORES  = {q["query"]: q["manual_scores"]  for q in QUERIES}
GROUND_TRUTH_TITLES = {q["query"]: q["gt_titles"] for q in QUERIES}
