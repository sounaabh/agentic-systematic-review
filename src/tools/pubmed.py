"""PubMed via NCBI E-utilities (free). Supports full Boolean + MeSH queries."""
import os
import xml.etree.ElementTree as ET

from src.state import Record
from src.tools.http import cached_get

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query: str, max_results: int = 100, year_from: int | None = None) -> list[Record]:
    common = {"db": "pubmed", "tool": "sr-agent"}
    if os.getenv("NCBI_EMAIL"):
        common["email"] = os.getenv("NCBI_EMAIL")
    term = f"({query}) AND {year_from}:3000[dp]" if year_from else query

    ids = cached_get(f"{EUTILS}/esearch.fcgi", params={
        **common, "term": term, "retmax": max_results, "retmode": "json",
    })["esearchresult"]["idlist"]
    if not ids:
        return []

    xml = cached_get(f"{EUTILS}/efetch.fcgi", params={
        **common, "id": ",".join(ids), "retmode": "xml",
    }, as_json=False)

    records = []
    for art in ET.fromstring(xml).findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join("".join(t.itertext()) for t in art.findall(".//AbstractText"))
        doi = next((e.text for e in art.findall(".//ArticleId") if e.get("IdType") == "doi"), None)
        year = art.findtext(".//PubDate/Year")
        authors = [f"{a.findtext('LastName', '')} {a.findtext('Initials', '')}".strip()
                   for a in art.findall(".//Author")[:10]]
        records.append(Record(
            id=f"doi:{doi.lower()}" if doi else f"pmid:{pmid}",
            title=title, abstract=abstract, authors=authors,
            year=int(year) if year and year.isdigit() else None,
            doi=doi, pmid=pmid, venue=art.findtext(".//Journal/Title"),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", sources=["pubmed"],
        ))
    return records
