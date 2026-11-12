"""Hybrid retrieval over one study's chunks: BM25 (sparse) + embeddings (dense),
fused with Reciprocal Rank Fusion. Every retrieved chunk keeps its id, so each
extracted value can be traced back to the exact passage it came from.
"""
import math
import re
from collections import Counter

from src.config import CONFIG

_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [_tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = _tok(query)
        out = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


class _Embedder:
    """Lazy Ollama embeddings; disabled automatically if unavailable."""
    _instance = None
    _failed = False

    @classmethod
    def get(cls):
        model = CONFIG["llm"].get("embedding_model")
        if cls._failed or not model or CONFIG["llm"]["provider"] != "ollama":
            return None
        if cls._instance is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                cls._instance = OllamaEmbeddings(model=model)
                cls._instance.embed_query("test")
            except Exception as e:  # model not pulled, Ollama not running...
                print(f"[rag] dense retrieval disabled ({e}); using BM25 only")
                cls._failed, cls._instance = True, None
        return cls._instance


def _cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class StudyIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.bm25 = BM25([c["text"] for c in chunks])
        emb = _Embedder.get()
        self.vectors = emb.embed_documents([c["text"] for c in chunks]) if emb and chunks else None

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self.chunks:
            return []
        sparse = self.bm25.scores(query)
        rankings = [sorted(range(len(self.chunks)), key=lambda i: -sparse[i])]
        if self.vectors:
            qv = _Embedder.get().embed_query(query)
            dense = [_cos(qv, v) for v in self.vectors]
            rankings.append(sorted(range(len(self.chunks)), key=lambda i: -dense[i]))
        fused = Counter()
        for ranking in rankings:                      # Reciprocal Rank Fusion
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (60 + rank)
        return [self.chunks[i] for i, _ in fused.most_common(k)]

    def multi_search(self, queries: list[str], k_total: int = 8) -> list[dict]:
        seen, out = set(), []
        per_q = max(2, k_total // len(queries) + 1)
        for q in queries:
            for c in self.search(q, per_q):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    out.append(c)
        return sorted(out[:k_total], key=lambda c: int(c["id"].split("-c")[-1]))
