import numpy as np
from search import BookSearchEngine

# two ground-truth books per query, automatically scored as 2 (highly relevant)
GROUND_TRUTH = {
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

# manual relevance scores for the top-5 results of each query and method
# 2 = highly relevant, 1 = somewhat relevant, 0 = not relevant
# ground-truth matches are auto-scored as 2, so you only need to judge the rest
# run the file once first to see which titles were retrieved, then fill these in
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


def dcg(scores):
    """Compute DCG for a ranked list of relevance scores."""
    # rank is 0-indexed, so we use rank+2 to get log2(2) at rank 0
    return sum(score / np.log2(rank + 2) for rank, score in enumerate(scores))


def ndcg(scores):
    """Normalize DCG against the ideal ranking."""
    ideal_dcg = dcg(sorted(scores, reverse=True))
    if ideal_dcg == 0:
        return 0.0
    return dcg(scores) / ideal_dcg


def is_ground_truth(title, gt_titles):
    """Case-insensitive substring check against the ground-truth titles."""
    title_lower = title.lower()
    return any(gt.lower() in title_lower or title_lower in gt.lower() for gt in gt_titles)


def print_results_for_grading(engine, n=5):
    """Print retrieved titles for all queries so manual scores can be assigned above."""
    print("=" * 60)
    print("RETRIEVED RESULTS — assign scores in MANUAL_SCORES")
    print("=" * 60)
    for query, gt_titles in GROUND_TRUTH.items():
        print(f"\n> {query}")
        print(f"  ground truth: {', '.join(gt_titles)}")
        for method, results in [
            ("bm25",     engine.bm25_search(query, n=n)),
            ("semantic", engine.semantic_search(query, top_k=n)),
        ]:
            print(f"\n  {method}:")
            for i, doc in enumerate(results, start=1):
                label = "[auto=2]" if is_ground_truth(doc["title"], gt_titles) else "[grade me: 0/1/2]"
                print(f"    rank {i}: {doc['title']} {label}")


def precision_at_k(results, gt_titles, k):
    """Fraction of the top-k results that are ground-truth books."""
    hits = sum(1 for doc in results[:k] if is_ground_truth(doc["title"], gt_titles))
    return hits / k


def recall_at_k(results, gt_titles, k):
    """Fraction of ground-truth books that appear in the top-k results."""
    hits = sum(1 for doc in results[:k] if is_ground_truth(doc["title"], gt_titles))
    return hits / len(gt_titles)


def evaluate(engine, n=5):
    """Compute nDCG@5, Precision@5, and Recall@5 for BM25 and semantic search across all queries."""
    all_ndcg      = {"bm25": [], "semantic": []}
    all_precision = {"bm25": [], "semantic": []}
    all_recall    = {"bm25": [], "semantic": []}

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for query, gt_titles in GROUND_TRUTH.items():
        print(f"\n> {query}")
        for method, results in [
            ("bm25",     engine.bm25_search(query, n=n)),
            ("semantic", engine.semantic_search(query, top_k=n)),
        ]:
            manual = MANUAL_SCORES[query][method]
            if manual is None:
                print(f"  {method}: skipped (no manual scores entered yet)")
                continue

            # use auto=2 for ground-truth matches, manual score for everything else
            scores = [
                2 if is_ground_truth(doc["title"], gt_titles) else manual[i]
                for i, doc in enumerate(results)
            ]

            ndcg_score = ndcg(scores)
            p_score    = precision_at_k(results, gt_titles, k=n)
            r_score    = recall_at_k(results, gt_titles, k=n)

            all_ndcg[method].append(ndcg_score)
            all_precision[method].append(p_score)
            all_recall[method].append(r_score)

            print(f"  {method:8s}  nDCG@{n}: {ndcg_score:.4f}  "
                  f"P@{n}: {p_score:.4f}  R@{n}: {r_score:.4f}  "
                  f"(scores: {scores})")

    print("\n" + "-" * 60)
    for method in ["bm25", "semantic"]:
        if all_ndcg[method]:
            print(f"  {method:8s}  "
                  f"mean nDCG@{n}: {np.mean(all_ndcg[method]):.4f}  "
                  f"mean P@{n}: {np.mean(all_precision[method]):.4f}  "
                  f"mean R@{n}: {np.mean(all_recall[method]):.4f}")
    print("-" * 60)


if __name__ == "__main__":
    engine = BookSearchEngine()

    # run once to see which titles were retrieved, then fill in MANUAL_SCORES above
    print_results_for_grading(engine, n=5)

    # once scores are filled in, this computes nDCG (skips queries with None)
    evaluate(engine, n=5)
