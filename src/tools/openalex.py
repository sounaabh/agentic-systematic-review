"""OpenAlex search (free, no key). Docs: https://docs.openalex.org"""
from src.state import Record
from src.tools.http import cached_get

BASE = "https://api.openalex.org/works"


def _abstract(inv_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inv_index:
        return ""
    words = sorted((pos, w) for w, positions in inv_index.items() for pos in positions)
    return " ".join(w for _, w in words)


def search_openalex(query: str, max_results: int = 100, year_from: int | None = None) -> list[Record]:
    filters = ["type:article"]
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    records, cursor = [], "*"
    while len(records) < max_results and cursor:
        data = cached_get(BASE, params={
            "search": query, "filter": ",".join(filters),
            "per-page": min(100, max_results), "cursor": cursor,
        })
        for w in data["results"]:
            doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
            oa = (w.get("best_oa_location") or {}) or {}
            records.append(Record(
                id=f"doi:{doi.lower()}" if doi else f"openalex:{w['id'].split('/')[-1]}",
                title=w.get("title") or "",
                abstract=_abstract(w.get("abstract_inverted_index")),
                authors=[a["author"]["display_name"] for a in w.get("authorships", [])[:10]],
                year=w.get("publication_year"),
                doi=doi,
                pmid=(w.get("ids", {}).get("pmid") or "").split("/")[-1] or None,
                venue=((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                url=w.get("id"),
                oa_pdf_url=oa.get("pdf_url"),
                sources=["openalex"],
            ))
        cursor = data["meta"].get("next_cursor")
    return records[:max_results]
