"""arXiv API search (free, no key). Good coverage for AI / engineering topics."""
import re
import xml.etree.ElementTree as ET

from src.state import Record
from src.tools.http import cached_get

BASE = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _to_arxiv_query(query: str) -> str:
    """'generative AI creativity' -> all:generative AND all:AI AND all:creativity.
    Quoted phrases are kept together: '"generative AI" creativity' -> all:"generative AI" AND all:creativity"""
    terms = re.findall(r'"[^"]+"|\S+', query)
    terms = [t for t in terms if t.upper() not in {"AND", "OR", "NOT"}]
    return " AND ".join(f"all:{t}" for t in terms)


def parse_arxiv_feed(xml: str, year_from: int | None = None) -> list[Record]:
    records = []
    for e in ET.fromstring(xml).findall("a:entry", NS):
        raw_id = e.findtext("a:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])
        year = int(e.findtext("a:published", default="0000", namespaces=NS)[:4] or 0) or None
        if year_from and year and year < year_from:
            continue
        doi = e.findtext("arxiv:doi", default=None, namespaces=NS)
        title = " ".join(e.findtext("a:title", default="", namespaces=NS).split())
        records.append(Record(
            id=f"doi:{doi.lower()}" if doi else f"arxiv:{arxiv_id}",
            title=title,
            abstract=" ".join(e.findtext("a:summary", default="", namespaces=NS).split()),
            authors=[a.findtext("a:name", default="", namespaces=NS) for a in e.findall("a:author", NS)][:10],
            year=year, doi=doi, arxiv_id=arxiv_id, venue="arXiv",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            oa_pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            sources=["arxiv"],
        ))
    return records


def search_arxiv(query: str, max_results: int = 40, year_from: int | None = None) -> list[Record]:
    xml = cached_get(BASE, params={
        "search_query": _to_arxiv_query(query), "start": 0,
        "max_results": max_results, "sortBy": "relevance",
    }, as_json=False)
    return parse_arxiv_feed(xml, year_from)
