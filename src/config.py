"""Loads configs/config.yaml and .env once, shared by the whole project.

Each team member can override the LLM settings in their own .env file
(never committed), so the shared config.yaml stays the same for everyone.
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Personal overrides from .env
    overrides = {"LLM_PROVIDER": "provider", "LLM_MODEL": "model",
                 "SCREENER_B_MODEL": "screener_b_model", "EMBEDDING_MODEL": "embedding_model"}
    for env_name, key in overrides.items():
        if os.getenv(env_name) is not None:
            cfg["llm"][key] = os.getenv(env_name)
    return cfg


CONFIG = load_config()
