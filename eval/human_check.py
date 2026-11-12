"""Human-vs-AI evaluation (accuracy, completeness, time savings).

Step 1 - export blind checking sheets for a finished run:
    python eval/human_check.py export runs/20261112-155525

    Creates in eval/:
      <run>_screening_check.xlsx   30 records (title + abstract), AI decisions hidden.
                                   Each human reviewer fills their own column
                                   (include / exclude) WITHOUT looking at the other.
      <run>_extraction_check.xlsx  extracted values of 5 studies next to the passages
                                   they came from; mark each value yes / partly / no.
    Write how many minutes each task took in the 'Instructions' sheets.

Step 2 - score:
    python eval/human_check.py score runs/20261112-155525
    -> prints the metrics and saves eval/<run>_evaluation.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openpyxl import Workbook, load_workbook                       # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill            # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation       # noqa: E402

from src.metrics import cohens_kappa                                # noqa: E402

EVAL = Path(__file__).resolve().parent
FIELDS = ["study_type", "sample_or_dataset", "ai_method", "outcomes", "key_findings", "effect_direction"]
HEAD = Font(bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="33415C")
WRAP = Alignment(wrap_text=True, vertical="top")


def load(run: Path, name: str):
    return json.loads((run / name).read_text(encoding="utf-8"))


def _header(ws, cols, widths):
    ws.append(cols)
    for i, w in enumerate(widths, 1):
        ws.cell(1, i).font, ws.cell(1, i).fill = HEAD, FILL
        ws.column_dimensions[chr(64 + i)].width = w
    ws.freeze_panes = "A2"


def _instructions(wb, lines, time_labels):
    ws = wb.create_sheet("Instructions", 0)
    ws.column_dimensions["A"].width = 60
    ws.column_dimensions["B"].width = 18
    for line in lines:
        ws.append([line])
    ws.append([])
    for label in time_labels:
        ws.append([label, ""])
        ws.cell(ws.max_row, 2).fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows():
        row[0].alignment = Alignment(wrap_text=True)


# --------------------------------------------------------------------------- export
def export(run: Path, n: int = 30):
    protocol = load(run, "protocol.json")
    records = {r["id"]: r for r in load(run, "records.json")}
    screening = load(run, "screening.json")

    # Stratified sample: half AI-included, half AI-excluded (otherwise almost no includes)
    rng = random.Random(42)
    inc = [s for s in screening if s["final"] == "include"]
    exc = [s for s in screening if s["final"] != "include"]
    k = min(len(inc), n // 2)
    sample = rng.sample(inc, k) + rng.sample(exc, min(len(exc), n - k))
    rng.shuffle(sample)

    wb = Workbook()
    ws = wb.active
    ws.title = "Screening"
    _header(ws, ["#", "record_id", "Title", "Abstract", "human_1", "human_2"], [5, 14, 40, 90, 12, 12])
    dv = DataValidation(type="list", formula1='"include,exclude"', allow_blank=True)
    ws.add_data_validation(dv)
    for i, s in enumerate(sample, 1):
        r = records[s["record_id"]]
        ws.append([i, r["id"], r["title"], r.get("abstract") or "(no abstract)", "", ""])
        for c in range(1, 7):
            ws.cell(i + 1, c).alignment = WRAP
        ws.row_dimensions[i + 1].height = 150
        dv.add(f"E{i + 1}:F{i + 1}")
    crit = (["INCLUSION:"] + [f"  - {c}" for c in protocol["inclusion_criteria"]]
            + ["EXCLUSION:"] + [f"  - {c}" for c in protocol["exclusion_criteria"]])
    _instructions(wb, [
        f"Review question: {protocol['research_question']}",
        "Screen each record on title + abstract with the criteria below.",
        "Each reviewer fills ONLY their own column (human_1 or human_2) with include / exclude,",
        "without looking at the other column. The AI decisions are hidden on purpose.",
        "Time yourself and write the minutes below.", ""] + crit,
        ["Minutes spent by human_1", "Minutes spent by human_2"])
    out1 = EVAL / f"{run.name}_screening_check.xlsx"
    wb.save(out1)

    # Extraction check: up to 5 confirmed studies, full-text first
    studies = {s["key"]: s for s in load(run, "evidence_chunks.json")}
    ex = [e for e in load(run, "extractions.json") if e.get("eligible_confirmed", True)]
    ex.sort(key=lambda e: not e.get("full_text"))
    wb = Workbook()
    ws = wb.active
    ws.title = "Extraction"
    _header(ws, ["Study", "Field", "Extracted value", "Evidence passages", "correct"], [8, 18, 50, 100, 12])
    dv = DataValidation(type="list", formula1='"yes,partly,no"', allow_blank=True)
    ws.add_data_validation(dv)
    for e in ex[:5]:
        chunks = {c["id"]: c["text"] for c in studies[e["key"]]["evidence_chunks"]}
        evidence = "\n\n".join(f"[{cid}] {chunks.get(cid, '')}" for cid in e.get("evidence", []))[:6000]
        for f in FIELDS:
            v = e.get(f)
            ws.append([e["key"], f, ", ".join(v) if isinstance(v, list) else str(v), evidence, ""])
            r = ws.max_row
            for c in range(1, 6):
                ws.cell(r, c).alignment = WRAP
            ws.row_dimensions[r].height = 120
            dv.add(f"E{r}")
    _instructions(wb, [
        "For each extracted value, read the evidence passages (and the PDF if needed).",
        "Mark 'yes' if the value is correct and complete, 'partly' if incomplete or",
        "imprecise, 'no' if wrong or not supported by the passages.",
        "Time yourself and write the minutes below."], ["Minutes spent checking"])
    out2 = EVAL / f"{run.name}_extraction_check.xlsx"
    wb.save(out2)
    print(f"Created:\n  {out1}\n  {out2}\nFill them in Excel, save, then run the 'score' command.")


# --------------------------------------------------------------------------- score
def _minutes(wb, label):
    for row in wb["Instructions"].iter_rows(values_only=True):
        if row and row[0] == label and row[1] not in (None, ""):
            try:
                return float(str(row[1]).replace(",", "."))
            except ValueError:
                return None
    return None


def _binary(stats_ref, stats_pred):
    tp = sum(r == "include" and p == "include" for r, p in zip(stats_ref, stats_pred))
    tn = sum(r == "exclude" and p == "exclude" for r, p in zip(stats_ref, stats_pred))
    fp = sum(r == "exclude" and p == "include" for r, p in zip(stats_ref, stats_pred))
    fn = sum(r == "include" and p == "exclude" for r, p in zip(stats_ref, stats_pred))
    n = len(stats_ref)
    return {"n": n, "accuracy": round((tp + tn) / n, 3) if n else None,
            "recall_sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "kappa": cohens_kappa(stats_ref, stats_pred) if n else None,
            "missed_includes": fn, "false_includes": fp}


def score(run: Path):
    res = {"run": run.name}
    screening = {s["record_id"]: s for s in load(run, "screening.json")}
    metrics = load(run, "metrics.json")
    b = lambda d: "exclude" if d == "exclude" else "include"   # 'uncertain' counts as include

    f1 = EVAL / f"{run.name}_screening_check.xlsx"
    if f1.exists():
        wb = load_workbook(f1)
        rows = [r for r in wb["Screening"].iter_rows(min_row=2, values_only=True) if r[1]]
        h1 = [(r[1], str(r[4]).strip().lower()) for r in rows if r[4]]
        h2 = {r[1]: str(r[5]).strip().lower() for r in rows if r[5]}
        # Reference standard: human consensus; disagreements resolved as include (conservative)
        ref = {rid: ("include" if "include" in (d, h2.get(rid, d)) else "exclude") for rid, d in h1}
        ids = list(ref)
        out = {"reference": "human consensus (disagreements counted as include)", "records_checked": len(ids)}
        out["ai_final_vs_humans"] = _binary([ref[i] for i in ids], [screening[i]["final"] for i in ids])
        out["ai_reviewer_A_vs_humans"] = _binary([ref[i] for i in ids], [b(screening[i]["reviewer_a"]["decision"]) for i in ids])
        out["ai_reviewer_B_vs_humans"] = _binary([ref[i] for i in ids], [b(screening[i]["reviewer_b"]["decision"]) for i in ids])
        both = [i for i, _ in h1 if i in h2]
        if both:
            d1 = dict(h1)
            out["human_1_vs_human_2_kappa"] = cohens_kappa([d1[i] for i in both], [h2[i] for i in both])
        mins = [m for m in (_minutes(wb, "Minutes spent by human_1"), _minutes(wb, "Minutes spent by human_2")) if m]
        if mins:
            per_abs = sum(mins) / len(mins) / max(1, len(ids))
            screened = metrics["prisma"].get("screened", 0)
            out["time"] = {
                "human_minutes_per_abstract": round(per_abs, 2),
                "estimated_manual_dual_screening_minutes": round(per_abs * screened * 2, 1),
                "records_screened_by_ai": screened,
                "ai_full_pipeline_minutes": metrics["metrics"].get("runtime_minutes"),
            }
        res["screening"] = out

    f2 = EVAL / f"{run.name}_extraction_check.xlsx"
    if f2.exists():
        wb = load_workbook(f2)
        marks = [str(r[4]).strip().lower() for r in wb["Extraction"].iter_rows(min_row=2, values_only=True) if r[4]]
        if marks:
            res["extraction_accuracy"] = {
                "values_checked": len(marks),
                "correct": round(marks.count("yes") / len(marks), 3),
                "partly": round(marks.count("partly") / len(marks), 3),
                "wrong": round(marks.count("no") / len(marks), 3),
                "minutes_spent_checking": _minutes(wb, "Minutes spent checking"),
            }

    # Completeness of the evidence table (no human input needed)
    ex = [e for e in load(run, "extractions.json") if e.get("eligible_confirmed", True)]
    vals = [e.get(f) for e in ex for f in FIELDS]
    filled = [v for v in vals if v and "not reported" not in str(v).lower() and str(v).lower() != "unclear"]
    res["completeness"] = {
        "evidence_table_fields_filled": round(len(filled) / len(vals), 3) if vals else None,
        "studies_with_full_text": f"{sum(bool(e.get('full_text')) for e in ex)}/{len(ex)}",
        "traceability_rate": metrics["metrics"].get("traceability_rate"),
        "inter_agent_kappa": metrics["metrics"].get("screening_kappa"),
    }
    out = EVAL / f"{run.name}_evaluation.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("export", "score"):
        print(__doc__)
        sys.exit(1)
    {"export": export, "score": score}[sys.argv[1]](Path(sys.argv[2]))
