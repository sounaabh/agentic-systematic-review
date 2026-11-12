"""Redo the last steps of a finished run without repeating search and screening.

    python resynthesize.py runs/20261112-154058
        -> synthesis -> critic -> report again (e.g. after improving the prompts)

    python resynthesize.py runs/20261112-154058 --report-only
        -> rebuilds review.md from the saved synthesis and critic log (no LLM calls)

    python resynthesize.py runs/20261112-154058 --extract-remaining
        -> first extracts the included studies that were skipped because of the
           max_studies cap, then synthesis -> critic -> report

Uses few LLM tokens compared with a full run. The report in the same run folder is
overwritten; the audit log keeps the history of every attempt.
"""
import argparse
import json
from pathlib import Path

from src.agents.extraction import extract_one, summarize_extraction
from src.agents.synthesis import critic_agent, route_after_critic, synthesis_agent
from src.audit import log, save_json
from src.llm import get_llm
from src.report.manuscript import report_agent


def load(run: Path, name: str):
    return json.loads((run / name).read_text(encoding="utf-8"))


def included_after_human(run: Path) -> list[str]:
    """Included ids after the human checkpoint (rebuilt from the audit log for older runs)."""
    if (run / "included_ids.json").exists():
        return load(run, "included_ids.json")
    ids = [s["record_id"] for s in load(run, "screening.json") if s["final"] == "include"]
    dropped = set()
    for line in (run / "audit.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e["event"] == "screening_override":
            dropped |= set(e["data"].get("excluded_by_human", []))
    return [i for i in ids if i not in dropped]


def saved_synthesis(run: Path) -> dict:
    """Last synthesis draft + the critic reviews of that synthesis round, from the audit log."""
    events = [json.loads(l) for l in (run / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [i for i, e in enumerate(events)
              if e["agent"] == "synthesis" and e["event"] == "draft_written" and e["data"].get("revision") == 0]
    last_round = events[starts[-1]:] if starts else events
    critic = [e["data"] for e in last_round if e["agent"] == "critic" and e["event"] == "review"]
    rev = max((e["data"]["revision"] for e in last_round
               if e["agent"] == "synthesis" and e["event"] == "draft_written"), default=0)
    text = load(run, f"synthesis_v{rev}.json")["text"]
    return {"synthesis": text, "critic_feedback": critic, "revisions": rev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--extract-remaining", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild the report, no LLM calls")
    args = ap.parse_args()

    run = Path(args.run)
    saved = load(run, "metrics.json")
    state = {
        "run_id": run.name,
        "protocol": load(run, "protocol.json"),
        "search_strategy": load(run, "search_strategy.json"),
        "extractions": load(run, "extractions.json"),
        "quality": load(run, "quality.json"),
        "studies": load(run, "evidence_chunks.json"),
        "metrics": saved["metrics"],
        "prisma": saved["prisma"],
    }
    log(state["run_id"], "orchestrator", "resynthesis_started", {"extract_remaining": args.extract_remaining})

    if args.extract_remaining:
        included = included_after_human(run)
        done = {s["record"]["id"] for s in state["studies"]}
        records = {r["id"]: r for r in load(run, "records.json")}
        todo = [records[i] for i in included if i not in done and i in records]
        print(f"Extracting {len(todo)} remaining included studies...")
        llm, q = get_llm(json_mode=True), state["protocol"]["research_question"]
        for rec in todo:
            st, ex, qa = extract_one(rec, f"S{len(state['studies']) + 1}", llm, q, state["run_id"])
            state["studies"].append(st); state["extractions"].append(ex); state["quality"].append(qa)
        state["prisma"], state["metrics"] = summarize_extraction(
            state["prisma"], state["metrics"], state["studies"], state["extractions"], len(included))
        save_json(state["run_id"], "extractions.json", state["extractions"])
        save_json(state["run_id"], "quality.json", state["quality"])
        save_json(state["run_id"], "evidence_chunks.json", state["studies"])

    if args.report_only:
        state.update(saved_synthesis(run))
        state.update(report_agent(state))
        print(f"Report rebuilt: {state['report_path']}")
        return

    while True:
        state.update(synthesis_agent(state))
        state.update(critic_agent(state))
        if route_after_critic(state) == "report":
            break
    state.update(report_agent(state))
    save_json(state["run_id"], "metrics.json", {"metrics": state["metrics"], "prisma": state["prisma"]})
    print(f"Report rewritten: {state['report_path']}")


if __name__ == "__main__":
    main()
