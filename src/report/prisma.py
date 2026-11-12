"""PRISMA 2020 flow diagram drawn with matplotlib from the run's counts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAMES = {"openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv", "pubmed": "PubMed"}


def _box(ax, x, y, w, h, text, color="#eef3fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.01",
                                fc=color, ec="#33415c", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#33415c", lw=1.2))


def draw_prisma(p: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    L, R, W, H = 0.17, 0.61, 0.37, 0.11
    rows = [0.84, 0.66, 0.48, 0.30, 0.10]

    per_source = "\n".join(f"{NAMES.get(k, k)} (n = {v})" for k, v in p.get("identified", {}).items())
    _box(ax, L, rows[0], W, H + 0.03, f"Records identified (n = {p.get('identified_total', 0)})\n{per_source}")
    _box(ax, R, rows[0], W, H, f"Records removed before screening:\n"
         f"Duplicates (n = {p.get('duplicates_removed', 0)})\n"
         f"Over screening cap (n = {p.get('not_screened_cap', 0)})", "#f7f7f7")
    _box(ax, L, rows[1], W, H, f"Records screened\n(title/abstract, 2 AI reviewers)\n(n = {p.get('screened', 0)})")
    _box(ax, R, rows[1], W, H, f"Records excluded\n(n = {p.get('excluded_screening', 0)})", "#f7f7f7")
    _box(ax, L, rows[2], W, H, f"Reports sought for retrieval\n(n = {p.get('reports_sought', 0)})")
    _box(ax, R, rows[2], W, H, f"Not assessed (over extraction cap)\n(n = {p.get('not_extracted_cap', 0)})\n"
         f"Full text not available, abstract used\n(n = {p.get('abstract_only', 0)})", "#f7f7f7")
    _box(ax, L, rows[3], W, H, f"Reports assessed for eligibility\n(n = {p.get('reports_sought', 0)})")
    _box(ax, R, rows[3], W, H, f"Reports excluded at full text\n(n = {p.get('excluded_fulltext', 0)})", "#f7f7f7")
    _box(ax, L, rows[4], W, H, f"Studies included in review\n(n = {p.get('included_final', 0)})", "#e3f4e8")

    for i in range(4):
        top_y = rows[i] + (0 if i else 0)
        _arrow(ax, L + W / 2, top_y, L + W / 2, rows[i + 1] + H)
        _arrow(ax, L + W, rows[i] + H / 2, R, rows[i] + H / 2)
    for y, label in zip([0.97, 0.79, 0.61, 0.43], ["Identification", "Screening", "Retrieval", "Eligibility"]):
        ax.text(0.01, y, label, fontsize=9, fontweight="bold", color="#33415c")
    ax.text(0.01, 0.24, "Included", fontsize=9, fontweight="bold", color="#33415c")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
