"""Compare a run against a published review.

1. Make eval/gold_<name>.csv with one column 'doi' or 'title' listing the studies that
   the published review included (and that are in the years/sources we searched).
2. python eval/evaluate.py runs/<run_id> eval/gold_<name>.csv

Reports screening recall / precision. Titles are matched fuzzily when no DOI is given.
"""
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def main(run, gold_csv):
    run = Path(run)
    records = {r["id"]: r for r in json.loads((run / "records.json").read_text(encoding="utf-8"))}
    screening = json.loads((run / "screening.json").read_text(encoding="utf-8"))
    included = [records[s["record_id"]] for s in screening if s["final"] == "include"]
    gold = list(csv.DictReader(open(gold_csv, encoding="utf-8")))

    def match(g, pool):
        if g.get("doi"):
            return any((r.get("doi") or "").lower() == g["doi"].strip().lower() for r in pool)
        return any(SequenceMatcher(None, norm(r["title"]), norm(g["title"])).ratio() > 0.9 for r in pool)

    retrieved = [g for g in gold if match(g, records.values())]
    found = [g for g in gold if match(g, included)]
    res = {
        "gold_studies": len(gold),
        "search_recall": round(len(retrieved) / len(gold), 3) if gold else None,
        "screening_recall_overall": round(len(found) / len(gold), 3) if gold else None,
        "screening_recall_of_retrieved": round(len(found) / len(retrieved), 3) if retrieved else None,
        "included_by_system": len(included),
        "precision_vs_gold": round(len(found) / len(included), 3) if included else None,
        "missed": [g.get("title") or g.get("doi") for g in gold if g not in found],
    }
    print(json.dumps(res, indent=2))
    (run / "evaluation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:3])
