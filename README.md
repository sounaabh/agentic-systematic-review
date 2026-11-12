# Agentic Systematic Review

A multi-agent AI pipeline that automates a **PRISMA 2020-aligned systematic literature review**,
from a research question to a draft review with a PRISMA flow diagram, built with **LangGraph**,
local LLMs (**Ollama**) and **hybrid RAG**.

## What it does

```
Topic
  │
  ▼
Protocol Agent ──► [human checkpoint: approve / edit protocol]
  │   PICOC question, inclusion & exclusion criteria
  ▼
Search Strategy Agent        concepts, synonyms, keyword queries
  ▼
Retrieval Agent              OpenAlex + Semantic Scholar + arXiv, deduplication
  ▼
Screening: Reviewer A ─┐
           Reviewer B ─┴─► Adjudicator ──► [human checkpoint: approve / remove]
  │   two independent LLM reviewers, Cohen's kappa
  ▼
Extraction + Quality Agents  open-access PDFs → chunks → hybrid RAG (BM25 + embeddings)
  │   structured data with passage-level evidence ids, MMAT-inspired appraisal
  ▼
Synthesis Agent ⇄ Critic Agent   narrative synthesis (SWiM), self-reflection loop
  ▼
Report Agent                 review.md, PRISMA diagram, tables (CSV), references.bib
```

| Requirement | How it is met |
|---|---|
| Multi-agent workflow | 8 LangGraph nodes sharing a typed `ReviewState` (`src/graph.py`) |
| Dual screening | Two different models/prompts screen independently; adjudicator resolves conflicts |
| RAG & traceability | Hybrid BM25 + dense retrieval with RRF; every extracted row cites chunk ids (`evidence_chunks.json`) |
| Human-in-the-loop | LangGraph `interrupt_after` checkpoints after protocol and screening |
| Reproducibility | temperature 0, fixed seed, cached API responses, config snapshot, JSONL audit log |
| Evaluation | Cohen's kappa, traceability rate, runtime, recall/precision vs. a published review (`eval/`) |

## Setup (Windows)

```powershell
git clone https://github.com/sounaabh/agentic-systematic-review.git
cd agentic-systematic-review
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), then pull the models (8 GB GPU is enough):

```powershell
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Check everything works:

```powershell
python check_setup.py
```

## Run a review

```powershell
python run.py --topic "effect of generative AI tools on human creativity and entrepreneurial competencies"
```

Add `--auto` to skip the human checkpoints. Settings (models, number of results, caps) are in
`configs/config.yaml`. To use a cloud model instead, set `provider: groq` and add your key to `.env`.

## Outputs (`runs/<run_id>/`)

| File | Content |
|---|---|
| `review.md` | The full draft review (methods, results, synthesis, limitations, references) |
| `prisma.png` | PRISMA 2020 flow diagram |
| `audit.jsonl` | Every agent decision with timestamp and model |
| `protocol.json`, `search_strategy.json` | Protocol and search queries |
| `screening.json` | Both reviewers' decisions, rationales and adjudications |
| `extractions.json`, `extraction_table.csv` | Extracted data with evidence ids |
| `evidence_chunks.json` | The exact passages used for each study |
| `quality.json` | Quality appraisal |
| `references.bib` | BibTeX of included studies |
| `metrics.json` | kappa, conflicts, traceability, runtime, PRISMA counts |

## Evaluation against a published review

Create `eval/gold_<name>.csv` with a `doi` (or `title`) column listing the studies included by a
published review on the same question, then:

```powershell
python eval/evaluate.py runs/<run_id> eval/gold_<name>.csv
```

## Case studies

1. **Generative AI, creativity and entrepreneurial competencies** (main demo)
2. **Deep reinforcement learning / adaptive AI for dynamic production planning** (generalisation test)

## Limitations

LLM decisions can be wrong and must be verified by humans; only open-access full texts are used;
the quality checklist is simplified; no meta-analysis is performed.

## Team

| Member | Contributions |
|---|---|
| Soundous | _to fill in_ |
| Naaela | _to fill in_ |
