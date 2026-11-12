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
1. Identify the key concepts of the protocol and list synonyms for each.
2. Write 3 to 4 SHORT keyword queries for high recall:
   - the FIRST query is the core of the question: intervention + main task or
     topic (e.g. "reinforcement learning dynamic scheduling"); never drop the main task,
   - each query has 2 to 4 words only (databases require EVERY word to match),
   - each query combines the intervention with ONE outcome or population concept,
   - if the question has several outcomes (e.g. creativity AND entrepreneurship),
     write separate queries for each instead of putting them in the same query,
   - no Boolean operators, no quotes, no field tags.
   Good: "generative AI creativity", "ChatGPT entrepreneurship students".
   Bad: "AI art generators originality business model development"."""


def search_strategy_agent(state: ReviewState) -> ReviewState:
    p = state["protocol"]
    user = (f"Research question: {p['research_question']}\nPopulation: {p['population']}\n"
            f"Intervention: {p['intervention']}\nOutcomes: {', '.join(p['outcomes'])}\nContext: {p['context']}")
    strategy = ask_structured(get_llm(json_mode=True), SearchStrategy, SYSTEM, user)
    strategy.queries = [q.strip().strip('"') for q in strategy.queries if q.strip()][:4]

    log(state["run_id"], "search_strategy", "strategy_generated", strategy.model_dump())
    save_json(state["run_id"], "search_strategy.json", strategy.model_dump())
    return {"search_strategy": strategy.model_dump()}


def retrieval_agent(state: ReviewState) -> ReviewState:
    cfg = CONFIG["search"]
    year_from = state["protocol"].get("year_from") or cfg.get("year_from")
    raw: list[Record] = []
    per_source: dict[str, list[Record]] = {}
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
            per_source.setdefault(source, []).extend(hits)
            log(state["run_id"], "retrieval", "search_executed", {"source": source, "query": q, "hits": len(hits)})
            print(f"  {source:17s} {len(hits):4d} hits  <- {q}")

    # Interleave sources round-robin (1st of each source, then 2nd of each...) so that
    # relevance-ranked results from every database survive the screening cap fairly.
    lists = list(per_source.values())
    for i in range(max((len(l) for l in lists), default=0)):
        raw += [l[i] for l in lists if i < len(l)]

    unique, n_dup = deduplicate(raw)
    cap = cfg["max_records_to_screen"]
    capped = unique[:cap]
    prisma = {"identified": identified, "identified_total": len(raw),
              "duplicates_removed": n_dup, "after_dedup": len(unique),
              "not_screened_cap": max(0, len(unique) - cap), "screened": len(capped)}

    log(state["run_id"], "retrieval", "deduplicated", prisma)
    save_json(state["run_id"], "records.json", [r.model_dump() for r in capped])
    print(f"  -> {len(raw)} records, {n_dup} duplicates removed, {len(capped)} to screen")
    return {"records": [r.model_dump() for r in capped], "prisma": prisma}
