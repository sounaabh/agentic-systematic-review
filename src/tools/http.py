"""HTTP GET with on-disk caching and polite retries.

Caching means re-running the pipeline returns the exact same search results,
which is important for reproducibility (and saves time while debugging).
"""
import hashlib
import json
import time
from pathlib import Path

import requests

from src.config import ROOT

CACHE_DIR = ROOT / "cache" / "http"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "sr-agent/0.1 (academic project)"}


def cached_get(url: str, params: dict | None = None, headers: dict | None = None,
               as_json: bool = True, retries: int = 3):
    key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["body"]

    for attempt in range(retries):
        r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=30)
        if r.status_code == 429:               # rate limited: back off and retry
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        body = r.json() if as_json else r.text
        path.write_text(json.dumps({"url": url, "params": params, "body": body}), encoding="utf-8")
        return body
    raise RuntimeError(f"Rate limited too many times: {url}")
