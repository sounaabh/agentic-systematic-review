"""Redo only the synthesis -> critic -> report steps on a finished run.

Useful after improving the synthesis/critic prompts: no new search, screening or
extraction, so it is fast and uses few LLM tokens. The new report overwrites
review.md in the same run folder (the audit log keeps the history of both attempts).

    python resynthesize.py runs/20261112-154058
"""
import json
import sys
from pathlib import Path

from src.agents.synthesis import critic_agent, route_after_critic, synthesis_agent
from src.audit import log, save_json
from src.report.manuscript import report_agent


def load(run: Path, name: str):
    return json.loads((run / name).read_text(encoding="utf-8"))


def main(run_path: str):
    run = Path(run_path)
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
    log(state["run_id"], "orchestrator", "resynthesis_started")
    while True:
        state.update(synthesis_agent(state))
        state.update(critic_agent(state))
        if route_after_critic(state) == "report":
            break
    state.update(report_agent(state))
    save_json(state["run_id"], "metrics.json", {"metrics": state["metrics"], "prisma": state["prisma"]})
    print(f"Report rewritten: {state['report_path']}")


if __name__ == "__main__":
    main(sys.argv[1])
