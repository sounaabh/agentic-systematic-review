"""Search Strategy Agent + Retrieval Agent."""
from src.audit import log, save_json
from src.config import CONFIG
from src.llm import ask_structured, get_llm
from src.state import Record, ReviewState, SearchStrategy
from src.tools.arxiv import search_arxiv
from src.tools.dedup import deduplicate
from src.tools.openalex import search_openalex
from src.tools.pubmed import search_pubmed
from src.tools.semantic_scholar import search_semantic_scholar

SOURCES = {
    "openalex": search_openalex,
    "semantic_scholar": search_semantic_scholar,
    "arxiv": search_arxiv,
    "pubmed": search_pubmed,
}

SYSTEM = """You are an information specialist designing the literature search for a systematic review.
1. Identify the 2-3 key concepts of the protocol and list synonyms for each.
2. Write 2 to 3 SHORT keyword queries (4-8 words, no Boolean operators, no field tags).
   Each query must combine at least two concepts. Together they should give high recall."""


def search_strategy_agent(state: ReviewState) -> ReviewState:
    p = state["protocol"]
    user = (f"Research question: {p['research_question']}\nPopulation: {p['population']}\n"
            f"Intervention: {p['intervention']}\nOutcomes: {', '.join(p['outcomes'])}\nContext: {p['context']}")
    strategy = ask_structured(get_llm(json_mode=True), SearchStrategy, SYSTEM, user)
    strategy.queries = [q.strip() for q in strategy.queries if q.strip()][:3]

    log(state["run_id"], "search_strategy", "strategy_generated", strategy.model_dump())
    save_json(state["run_id"], "search_strategy.json", strategy.model_dump())
    return {"search_strategy": strategy.model_dump()}


def retrieval_agent(state: ReviewState) -> ReviewState:
    cfg = CONFIG["search"]
    year_from = state["protocol"].get("year_from") or cfg.get("year_from")
    raw: list[Record] = []
    identified = {}

    for source in cfg["sources"]:
        identified[source] = 0
        for q in state["search_strategy"]["queries"]:
            try:
                hits = SOURCES[source](q, cfg["max_results_per_query"], year_from)
            except Exception as e:      # one failing database must not kill the review
                log(state["run_id"], "retrieval", "source_error", {"source": source, "query": q, "error": str(e)})
                print(f"  ! {source} failed for '{q}': {e}")
                continue
            identified[source] += len(hits)
            raw += hits
            log(state["run_id"], "retrieval", "search_executed", {"source": source, "query": q, "hits": len(hits)})
            print(f"  {source:17s} {len(hits):4d} hits  <- {q}")

    unique, n_dup = deduplicate(raw)
    # Interleave sources so the cap keeps a fair mix, then cap
    cap = cfg["max_records_to_screen"]
    capped = unique[:cap]
    prisma = {"identified": identified, "identified_total": len(raw),
              "duplicates_removed": n_dup, "after_dedup": len(unique),
              "not_screened_cap": max(0, len(unique) - cap), "screened": len(capped)}

    log(state["run_id"], "retrieval", "deduplicated", prisma)
    save_json(state["run_id"], "records.json", [r.model_dump() for r in capped])
    print(f"  -> {len(raw)} records, {n_dup} duplicates removed, {len(capped)} to screen")
    return {"records": [r.model_dump() for r in capped], "prisma": prisma}
