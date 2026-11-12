"""Full-text retrieval + RAG-based Data Extraction Agent + Quality Appraisal Agent.

For each included study: get the open-access PDF (or fall back to the abstract),
chunk it, retrieve the most relevant chunks with hybrid search, and ask the LLM
to extract a structured row. Every row cites the chunk ids it used, and those ids
are checked against the real chunks (traceability).
"""
from src.audit import log, save_json
from src.config import CONFIG
from src.llm import ask_structured, get_llm
from src.rag import StudyIndex, retrieval_mode
from src.state import Extraction, QualityAppraisal, ReviewState
from src.tools.fulltext import chunk_text, download_pdf, pdf_to_text

RAG_QUERIES = [
    "study design method participants sample size dataset",
    "results findings effect significant improvement compared",
    "limitations future research threats to validity",
]

EXTRACT_PROMPT = """You extract data for a systematic review. Use ONLY the passages given.
Each passage starts with its id in brackets, e.g. [S3-c4]. Put the ids of the passages you
used in "evidence". If a value is not reported in the passages, write "not reported".
Never invent numbers. Set eligible_confirmed=false only if the passages clearly show the
study does not meet the review question (then give exclusion_reason)."""

QUALITY_PROMPT = """You appraise study quality with a simplified checklist inspired by the
Mixed Methods Appraisal Tool (MMAT). Answer each item yes / no / unclear using ONLY the
passages, then give an overall rating (high, moderate, low) with a short justification."""


def _prepare_study(rec: dict, key: str) -> dict:
    cfg = CONFIG["fulltext"]
    pdf = download_pdf(rec.get("oa_pdf_url"))
    text, full = "", False
    if pdf:
        try:
            text, full = pdf_to_text(pdf), True
        except Exception:
            text = ""
    if len(text.split()) < 300:          # failed or empty parse -> abstract fallback
        text, full = f"{rec['title']}. {rec.get('abstract', '')}", False
    chunks = chunk_text(text, key, cfg["chunk_words"], cfg["chunk_overlap"])
    return {"key": key, "record": rec, "full_text": full, "chunks": chunks}


def extract_one(rec: dict, key: str, llm, question: str, run_id: str):
    """Full text -> chunks -> hybrid retrieval -> structured extraction + quality appraisal."""
    cfg = CONFIG["fulltext"]
    study = _prepare_study(rec, key)
    index = StudyIndex(study["chunks"])
    ctx = index.multi_search(RAG_QUERIES + [question], cfg["top_k_chunks"])
    passages = "\n\n".join(f"[{c['id']}] {c['text']}" for c in ctx)
    user = f"Review question: {question}\nStudy: {rec['title']} ({rec.get('year')})\n\nPASSAGES:\n{passages}"

    try:
        ex = ask_structured(llm, Extraction, EXTRACT_PROMPT, user).model_dump()
    except Exception as e:
        ex = {"eligible_confirmed": True, "error": str(e), "evidence": [], "effect_direction": "unclear"}
    valid_ids = {c["id"] for c in ctx}
    ex["evidence"] = [e for e in ex.get("evidence", []) if e in valid_ids]   # drop hallucinated ids
    ex.update(key=key, record_id=rec["id"], title=rec["title"], full_text=study["full_text"])

    try:
        qa = ask_structured(llm, QualityAppraisal, QUALITY_PROMPT, user).model_dump()
    except Exception as e:
        qa = {"overall": "low", "justification": f"Appraisal failed: {e}"}
    qa.update(key=key, record_id=rec["id"])

    log(run_id, "extraction", "extracted", ex)
    log(run_id, "quality", "appraised", qa)
    print(f"  {key} {'PDF ' if study['full_text'] else 'ABS '} q={qa.get('overall'):8s} {rec['title'][:65]}")
    return ({"key": key, "record": rec, "full_text": study["full_text"], "evidence_chunks": ctx}, ex, qa)


def summarize_extraction(prisma: dict, metrics: dict, studies, extractions, n_included_screening: int):
    confirmed = [e for e in extractions if e.get("eligible_confirmed", True)]
    traced = [e for e in confirmed if e.get("evidence")]
    prisma = {**prisma,
              "reports_sought": len(studies),
              "not_extracted_cap": n_included_screening - len(studies),
              "full_text_retrieved": sum(s["full_text"] for s in studies),
              "abstract_only": sum(not s["full_text"] for s in studies),
              "excluded_fulltext": len(extractions) - len(confirmed),
              "included_final": len(confirmed)}
    metrics = {**metrics,
               "traceability_rate": round(len(traced) / len(confirmed), 3) if confirmed else None,
               "retrieval_mode": retrieval_mode()}
    return prisma, metrics


def extraction_agent(state: ReviewState) -> ReviewState:
    run_id = state["run_id"]
    by_id = {r["id"]: r for r in state["records"]}
    included = [by_id[i] for i in state["included_ids"]]
    # Prefer studies with an open-access PDF when we have to cap
    included.sort(key=lambda r: r.get("oa_pdf_url") is None)
    selected = included[: CONFIG["fulltext"]["max_studies"]]

    llm = get_llm(json_mode=True)
    q = state["protocol"]["research_question"]
    studies, extractions, quality = [], [], []
    for n, rec in enumerate(selected, 1):
        st, ex, qa = extract_one(rec, f"S{n}", llm, q, run_id)
        studies.append(st); extractions.append(ex); quality.append(qa)

    prisma, metrics = summarize_extraction(state["prisma"], state["metrics"], studies, extractions, len(included))
    save_json(run_id, "extractions.json", extractions)
    save_json(run_id, "quality.json", quality)
    save_json(run_id, "evidence_chunks.json", studies)
    save_json(run_id, "included_ids.json", state["included_ids"])   # after human checkpoint
    return {"studies": studies, "extractions": extractions, "quality": quality,
            "prisma": prisma, "metrics": metrics}
