"""Run a full systematic review.

    python run.py --topic "effect of generative AI on human creativity"
    python run.py --topic "..." --auto        # no human checkpoints
"""
import argparse
import json
import time

from src.audit import log, run_dir, save_json
from src.config import CONFIG
from src.graph import build_graph


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def checkpoint_protocol(graph, cfg, run_id):
    path = run_dir(run_id) / "protocol.json"
    proto = graph.get_state(cfg).values["protocol"]
    banner("HUMAN CHECKPOINT 1: review the protocol")
    print(json.dumps(proto, indent=2, ensure_ascii=False))
    print(f"\nYou can edit {path} now (criteria, outcomes...).")
    input("Press Enter to approve and continue... ")
    edited = json.loads(path.read_text(encoding="utf-8"))
    if edited != proto:
        graph.update_state(cfg, {"protocol": edited})
        log(run_id, "human", "protocol_edited", edited)
        print("  -> edited protocol saved")
    else:
        log(run_id, "human", "protocol_approved")


def checkpoint_screening(graph, cfg, run_id):
    st = graph.get_state(cfg).values
    inc = [s for s in st["screening"] if s["final"] == "include"]
    banner(f"HUMAN CHECKPOINT 2: {len(inc)} records included at screening "
           f"(kappa = {st['metrics'].get('screening_kappa')})")
    for i, s in enumerate(inc, 1):
        flag = " (adjudicated)" if s["adjudicated"] else ""
        print(f"  {i:3d}. {s['title'][:90]}{flag}")
    ans = input("\nNumbers to EXCLUDE (e.g. 3,7) or Enter to approve: ").strip()
    if ans:
        drop = {inc[int(x) - 1]["record_id"] for x in ans.replace(" ", "").split(",") if x.isdigit()
                and 0 < int(x) <= len(inc)}
        ids = [i for i in st["included_ids"] if i not in drop]
        graph.update_state(cfg, {"included_ids": ids})
        log(run_id, "human", "screening_override", {"excluded_by_human": sorted(drop)})
        print(f"  -> {len(drop)} removed by human reviewer")
    else:
        log(run_id, "human", "screening_approved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--auto", action="store_true", help="skip human checkpoints")
    args = ap.parse_args()

    hitl = CONFIG["workflow"]["human_in_the_loop"] and not args.auto
    run_id = time.strftime("%Y%m%d-%H%M%S")
    graph = build_graph(hitl)
    cfg = {"configurable": {"thread_id": run_id}, "recursion_limit": 50}
    save_json(run_id, "config_snapshot.json", CONFIG)
    log(run_id, "orchestrator", "run_started", {"topic": args.topic, "human_in_the_loop": hitl})

    t0 = time.time()
    state_in = {"run_id": run_id, "topic": args.topic, "metrics": {}, "prisma": {}}
    banner(f"Run {run_id}\nTopic: {args.topic}")
    while True:
        for update in graph.stream(state_in, cfg, stream_mode="updates"):
            for node in update:
                print(f"[done] {node}")
        nxt = graph.get_state(cfg).next
        if not nxt:
            break
        state_in = None   # resume from checkpoint
        done = graph.get_state(cfg).values
        if "screening" in done and "extractions" not in done:
            checkpoint_screening(graph, cfg, run_id)
        else:
            checkpoint_protocol(graph, cfg, run_id)

    final = graph.get_state(cfg).values
    final["metrics"]["runtime_minutes"] = round((time.time() - t0) / 60, 1)
    save_json(run_id, "metrics.json", {"metrics": final["metrics"], "prisma": final["prisma"]})
    banner("DONE")
    print(f"Report:  {final['report_path']}")
    print(f"Folder:  {run_dir(run_id)}")
    print(json.dumps(final["metrics"], indent=2))


if __name__ == "__main__":
    main()
