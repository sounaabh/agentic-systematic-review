"""Loads configs/config.yaml and .env once, shared by the whole project."""
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
