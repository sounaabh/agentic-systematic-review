"""Copy a finished run into results/<name>/ so it can be committed to GitHub.

(runs/ is git-ignored because it is a working folder; results/ is the curated archive.)

    python export_results.py runs/20261112-155525 creativity_groq
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(run: str, name: str):
    src, dst = Path(run), ROOT / "results" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for f in (ROOT / "eval").glob(f"{src.name}_*"):       # human checks + evaluation
        shutil.copy(f, dst / f.name)
    print(f"Copied {src} -> {dst}")
    print(f"Now: git add results/{name}  then commit and push")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
