"""Audit trail: every agent decision is appended to runs/<run_id>/audit.jsonl.

This is what makes the review transparent and reproducible: anyone can read
exactly what each agent decided, why, and with which model.
"""
import json
import time
from pathlib import Path

from src.config import CONFIG, ROOT


def run_dir(run_id: str) -> Path:
    d = ROOT / CONFIG["workflow"]["runs_dir"] / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(run_id: str, agent: str, event: str, data: dict | None = None, model: str | None = None) -> None:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent": agent,
        "event": event,
        "model": model or f"{CONFIG['llm']['provider']}:{CONFIG['llm']['model']}",
        "data": data or {},
    }
    with open(run_dir(run_id) / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_json(run_id: str, name: str, obj) -> Path:
    path = run_dir(run_id) / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
