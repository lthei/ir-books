import numpy as np
from search import BookSearchEngine
from queries import GROUND_TRUTH, MANUAL_SCORES


def dcg(scores):
    """Compute DCG for a ranked list of relevance scores."""
    return sum(score / np.log2(rank + 2) for rank, score in enumerate(scores))


def ndcg(scores):
    """Normalize DCG against the ideal ranking."""
    ideal_dcg = dcg(sorted(scores, reverse=True))
    if ideal_dcg == 0:
        return 0.0
    return dcg(scores) / ideal_dcg


def is_ground_truth(doc, gt_ids):
    """Check if a retrieved doc matches any ground-truth ID exactly."""
    return doc["id"] in gt_ids


def precision_at_k(results, gt_ids, k):
    """Fraction of the top-k results that are ground-truth books."""
    hits = sum(1 for doc in results[:k] if is_ground_truth(doc, gt_ids))
    return hits / k


def recall_at_k(results, gt_ids, k):
    """Fraction of ground-truth books that appear in the top-k results."""
    hits = sum(1 for doc in results[:k] if is_ground_truth(doc, gt_ids))
    return hits / len(gt_ids)


def print_results_for_grading(engine, n=5):
    """Print retrieved titles for all queries so manual scores can be assigned above."""
    print("=" * 60)
    print("RETRIEVED RESULTS — assign scores in MANUAL_SCORES")
    print("=" * 60)
    for query, gt_ids in GROUND_TRUTH.items():
        print(f"\n> {query}")
        print(f"  ground truth IDs: {', '.join(gt_ids)}")
        for method, results in [
            ("bm25",     engine.bm25_search(query, n=n)),
            ("semantic", engine.semantic_search(query, top_k=n)),
        ]:
            print(f"\n  {method}:")
            for i, doc in enumerate(results, start=1):
                label = "[auto=2]" if is_ground_truth(doc, gt_ids) else "[grade me: 0/1/2]"
                print(f"    rank {i}: {doc['title']} (id: {doc['id']}) {label}")


def evaluate(engine, n=5):
    """Compute nDCG@5, Precision@5, and Recall@5 for BM25 and semantic search across all queries."""
    all_ndcg      = {"bm25": [], "semantic": []}
    all_precision = {"bm25": [], "semantic": []}
    all_recall    = {"bm25": [], "semantic": []}

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for query, gt_ids in GROUND_TRUTH.items():
        print(f"\n> {query}")
        for method, results in [
            ("bm25",     engine.bm25_search(query, n=n)),
            ("semantic", engine.semantic_search(query, top_k=n)),
        ]:
            manual = MANUAL_SCORES[query][method]
            if manual is None:
                print(f"  {method}: skipped (no manual scores entered yet)")
                continue

            # auto-score ground-truth matches as 2, use manual score for everything else
            scores = [
                2 if is_ground_truth(doc, gt_ids) else manual[i]
                for i, doc in enumerate(results)
            ]

            ndcg_score = ndcg(scores)
            p_score    = precision_at_k(results, gt_ids, k=n)
            r_score    = recall_at_k(results, gt_ids, k=n)

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

    # step 1: run once to see retrieved titles, then fill in MANUAL_SCORES in queries.py
    print_results_for_grading(engine, n=5)

    # step 2: once scores are filled in, this computes all metrics (skips queries with None)
    evaluate(engine, n=5)