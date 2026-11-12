"""Evaluation metrics."""


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement between two screeners beyond chance (1 = perfect, 0 = chance)."""
    assert len(a) == len(b) and a, "need two equal, non-empty lists"
    labels = sorted(set(a) | set(b))
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if p_e == 1 else round((p_o - p_e) / (1 - p_e), 3)


def recall_precision(predicted: set[str], gold: set[str]) -> dict:
    """Compare included studies against a published review's included studies."""
    tp = len(predicted & gold)
    return {
        "recall": round(tp / len(gold), 3) if gold else None,
        "precision": round(tp / len(predicted), 3) if predicted else None,
        "true_positives": tp, "missed": sorted(gold - predicted),
    }
