"""Download open-access PDFs and turn them into text chunks.

Only open-access PDFs are used (arXiv, OpenAlex / Semantic Scholar OA links).
If no PDF can be obtained, the study falls back to its abstract, and the report
says so (PRISMA: 'reports not retrieved').
"""
import hashlib

import requests

from src.config import ROOT
from src.tools.http import HEADERS

PDF_DIR = ROOT / "cache" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


def download_pdf(url: str) -> bytes | None:
    if not url:
        return None
    path = PDF_DIR / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".pdf")
    if path.exists():
        return path.read_bytes()
    try:
        r = requests.get(url, headers=HEADERS, timeout=40, allow_redirects=True)
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            return None          # landing page / paywall, not a real PDF
        path.write_bytes(r.content)
        return r.content
    except requests.RequestException:
        return None


def pdf_to_text(data: bytes, max_pages: int = 30) -> str:
    import pymupdf  # pip install pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in list(doc)[:max_pages])
    # Drop the reference list: it adds noise to retrieval
    for marker in ("\nReferences\n", "\nREFERENCES\n", "\nBibliography\n"):
        idx = text.rfind(marker)
        if idx > len(text) * 0.5:
            text = text[:idx]
            break
    return text


def chunk_text(text: str, key: str, size: int = 220, overlap: int = 40) -> list[dict]:
    """Split into overlapping word windows. Each chunk gets a stable id like 'S3-c7'."""
    words = text.split()
    chunks, step = [], max(1, size - overlap)
    for i, start in enumerate(range(0, max(1, len(words)), step)):
        piece = words[start : start + size]
        if not piece:
            break
        chunks.append({"id": f"{key}-c{i}", "text": " ".join(piece)})
        if start + size >= len(words):
            break
    return chunks
