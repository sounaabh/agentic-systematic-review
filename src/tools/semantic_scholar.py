"""Semantic Scholar Graph API (free; a key in .env raises rate limits)."""
import os

from src.state import Record
from src.tools.http import cached_get

BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,authors,year,venue,externalIds,url,openAccessPdf"


def search_semantic_scholar(query: str, max_results: int = 100, year_from: int | None = None) -> list[Record]:
    headers = {"x-api-key": os.getenv("S2_API_KEY")} if os.getenv("S2_API_KEY") else None
    records, offset = [], 0
    while len(records) < max_results:
        params = {"query": query, "fields": FIELDS, "limit": min(100, max_results), "offset": offset}
        if year_from:
            params["year"] = f"{year_from}-"
        data = cached_get(BASE, params=params, headers=headers)
        batch = data.get("data", [])
        if not batch:
            break
        for p in batch:
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI")
            records.append(Record(
                id=f"doi:{doi.lower()}" if doi else f"s2:{p['paperId']}",
                title=p.get("title") or "", abstract=p.get("abstract") or "",
                authors=[a["name"] for a in (p.get("authors") or [])[:10]],
                year=p.get("year"), doi=doi, pmid=ext.get("PubMed"),
                venue=p.get("venue"), url=p.get("url"),
                oa_pdf_url=(p.get("openAccessPdf") or {}).get("url"),
                sources=["semantic_scholar"],
            ))
        offset += len(batch)
        if "next" not in data:
            break
    return records[:max_results]
