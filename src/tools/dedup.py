"""Deduplication across databases.

Two records are duplicates if they share a DOI or PMID, or if their normalized
titles are near-identical (same first author year not required: titles alone
catch most cases). Merged records keep the union of their metadata and sources.
"""
import re
from difflib import SequenceMatcher

from src.state import Record


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _merge(a: Record, b: Record) -> Record:
    data = a.model_dump()
    for k, v in b.model_dump().items():
        if k == "sources":
            data[k] = sorted(set(data[k]) | set(v))
        elif not data.get(k) and v:
            data[k] = v
    if len(b.abstract) > len(a.abstract):       # keep the longest abstract
        data["abstract"] = b.abstract
    return Record(**data)


def deduplicate(records: list[Record], title_threshold: float = 0.93) -> tuple[list[Record], int]:
    """Returns (unique_records, number_of_duplicates_removed)."""
    unique: list[Record] = []
    by_doi, by_pmid, by_title = {}, {}, {}

    for r in records:
        if not r.title:
            continue
        nt = _norm_title(r.title)
        match = None
        if r.doi and r.doi.lower() in by_doi:
            match = by_doi[r.doi.lower()]
        elif r.pmid and r.pmid in by_pmid:
            match = by_pmid[r.pmid]
        elif nt in by_title:
            match = by_title[nt]
        else:  # fuzzy title match, only against titles of similar length
            for t, idx in by_title.items():
                if abs(len(t) - len(nt)) < 15 and SequenceMatcher(None, t, nt).ratio() >= title_threshold:
                    match = idx
                    break

        if match is None:
            unique.append(r)
            match = len(unique) - 1
        else:
            unique[match] = _merge(unique[match], r)

        m = unique[match]
        if m.doi: by_doi[m.doi.lower()] = match
        if m.pmid: by_pmid[m.pmid] = match
        by_title[nt] = match

    return unique, len(records) - len(unique)
