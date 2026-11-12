"""Dual independent screening (Reviewer A, Reviewer B) + Adjudicator.

Mirrors the Cochrane practice of two independent reviewers: the two screeners use
different prompts (and, by default, different models) and never see each other's
answers. Disagreements and 'uncertain' cases go to an adjudicator. Agreement is
reported as Cohen's kappa.

Each screener processes ALL records before the next one starts, so Ollama only
loads each model once (much faster on an 8 GB GPU).
"""
from src.audit import log, save_json
from src.config import CONFIG
from src.llm import ask_structured, get_llm, model_name
from src.metrics import cohens_kappa
from src.state import Adjudication, ReviewState, ScreeningDecision

PROMPT_A = """You are Reviewer A screening titles and abstracts for a systematic review.
Apply the eligibility criteria strictly. Include only if the record plausibly meets ALL
inclusion criteria and NO exclusion criterion. If the abstract is missing or too vague
to decide, answer "uncertain"."""

PROMPT_B = """You are Reviewer B, an independent second screener for a systematic review.
Read the record, then check each criterion one by one before deciding.
At title/abstract stage, when in doubt lean towards "uncertain" rather than excluding
a potentially relevant study. Answer "exclude" only when a criterion is clearly violated."""

ADJ_PROMPT = """You are the senior reviewer resolving a screening disagreement.
Read the criteria, the record and both reviewers' reasoning, then give the final decision.
Include if the study plausibly meets the criteria and could be checked at full-text stage."""


def _criteria(p: dict) -> str:
    inc = "\n".join(f"  I{i+1}. {c}" for i, c in enumerate(p["inclusion_criteria"]))
    exc = "\n".join(f"  E{i+1}. {c}" for i, c in enumerate(p["exclusion_criteria"]))
    return f"Research question: {p['research_question']}\nINCLUSION:\n{inc}\nEXCLUSION:\n{exc}"


def _record_text(r: dict) -> str:
    abstract = r["abstract"][:2500] if r.get("abstract") else "(no abstract available)"
    return f"TITLE: {r['title']}\nYEAR: {r.get('year')}\nABSTRACT: {abstract}"


def _screen_all(records, prompt, model, criteria, run_id, label):
    llm = get_llm(model, json_mode=True)
    out = {}
    for i, r in enumerate(records, 1):
        try:
            d = ask_structured(llm, ScreeningDecision, f"{prompt}\n\n{criteria}", _record_text(r))
        except Exception as e:
            d = ScreeningDecision(decision="uncertain", criteria_triggered=[],
                                  rationale=f"LLM error: {e}"[:200], confidence=0)
        out[r["id"]] = d
        log(run_id, f"screener_{label}", "decision", {"record": r["id"], **d.model_dump()}, model_name(model))
        print(f"  [{label}] {i}/{len(records)} {d.decision:9s} {r['title'][:70]}")
    return out


def screening_agent(state: ReviewState) -> ReviewState:
    run_id, records = state["run_id"], state["records"]
    criteria = _criteria(state["protocol"])
    model_a = CONFIG["llm"]["model"]
    model_b = CONFIG["llm"].get("screener_b_model") or model_a

    a = _screen_all(records, PROMPT_A, model_a, criteria, run_id, "A")
    b = _screen_all(records, PROMPT_B, model_b, criteria, run_id, "B")

    adj_llm = get_llm(json_mode=True)
    results, n_conflicts = [], 0
    for r in records:
        da, db = a[r["id"]], b[r["id"]]
        if da.decision == db.decision and da.decision != "uncertain":
            final, adjudicated, why = da.decision, False, None
        else:
            n_conflicts += 1
            user = (f"{_record_text(r)}\n\nReviewer A: {da.decision} - {da.rationale}\n"
                    f"Reviewer B: {db.decision} - {db.rationale}")
            try:
                adj = ask_structured(adj_llm, Adjudication, f"{ADJ_PROMPT}\n\n{criteria}", user)
                final, why = adj.decision, adj.rationale
            except Exception as e:
                final, why = "include", f"Adjudication failed ({e}); kept for full-text check"
            adjudicated = True
            log(run_id, "adjudicator", "decision", {"record": r["id"], "final": final, "rationale": why})
        results.append({"record_id": r["id"], "title": r["title"],
                        "reviewer_a": da.model_dump(), "reviewer_b": db.model_dump(),
                        "final": final, "adjudicated": adjudicated, "adjudication_rationale": why})

    labels_a = [a[r["id"]].decision for r in records]
    labels_b = [b[r["id"]].decision for r in records]
    raw_agree = sum(x == y for x, y in zip(labels_a, labels_b)) / max(1, len(records))
    included = [x["record_id"] for x in results if x["final"] == "include"]

    metrics = {**state.get("metrics", {}),
               "screening_kappa": cohens_kappa(labels_a, labels_b) if records else None,
               "screening_raw_agreement": round(raw_agree, 3),
               "screening_conflicts": n_conflicts,
               "screener_a_model": model_name(model_a), "screener_b_model": model_name(model_b)}
    prisma = {**state["prisma"], "excluded_screening": len(records) - len(included),
              "included_screening": len(included)}

    save_json(run_id, "screening.json", results)
    print(f"  -> kappa={metrics['screening_kappa']}, conflicts={n_conflicts}, included={len(included)}")
    return {"screening": results, "included_ids": included, "metrics": metrics, "prisma": prisma}
