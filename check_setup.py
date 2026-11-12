"""Run this first: checks Python, libraries, Ollama models and the paper databases."""
import sys

print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "-> need 3.10+")

missing = []
for mod in ["langgraph", "langchain_ollama", "pydantic", "yaml", "requests", "dotenv", "pymupdf", "matplotlib"]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
print("Libraries:", "OK" if not missing else f"MISSING {missing} -> run: pip install -r requirements.txt")

from src.config import CONFIG

cfg = CONFIG["llm"]
for model in {cfg["model"], cfg.get("screener_b_model")} - {None, ""}:
    try:
        from src.llm import get_llm
        reply = get_llm(model).invoke("Reply with the single word: ready").content
        print(f"LLM {model}: OK ({reply.strip()[:30]})")
    except Exception as e:
        print(f"LLM {model}: FAILED -> is Ollama running? did you run 'ollama pull {model}'?  ({e})")

if cfg.get("embedding_model"):
    try:
        from langchain_ollama import OllamaEmbeddings
        OllamaEmbeddings(model=cfg["embedding_model"]).embed_query("test")
        print(f"Embeddings {cfg['embedding_model']}: OK")
    except Exception as e:
        print(f"Embeddings: FAILED -> run 'ollama pull {cfg['embedding_model']}' (optional)  ({e})")

from src.tools.arxiv import search_arxiv
from src.tools.openalex import search_openalex
from src.tools.semantic_scholar import search_semantic_scholar

for name, fn in [("OpenAlex", search_openalex), ("Semantic Scholar", search_semantic_scholar), ("arXiv", search_arxiv)]:
    try:
        hits = fn("generative AI creativity", 3, 2020)
        print(f"{name}: OK ({len(hits)} results, e.g. '{hits[0].title[:60]}')" if hits else f"{name}: 0 results")
    except Exception as e:
        print(f"{name}: FAILED ({e})")
