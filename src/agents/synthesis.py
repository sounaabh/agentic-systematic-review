"""Synthesis Agent + Critic Agent (self-reflection loop).

The synthesis agent writes a narrative synthesis citing studies as [S1], [S2]...
The critic checks it (LLM review + deterministic citation check). If problems are
found, the synthesis is revised, up to `max_critic_revisions` times.
"""
import re
from collections import Counter

from src.audit import log, save_json
from src.config import CONFIG
from src.llm import ask_structured, ask_text, get_llm
from src.state import Critique, ReviewState

SYNTH_PROMPT = """You write the narrative synthesis section of a systematic review (PRISMA 2020,
SWiM guideline). Use ONLY the study table provided.
Rules:
- Cite every factual statement with study keys in brackets, e.g. [S1] or [S2, S5].
- Group studies into 2-4 themes, then summarise the direction of effects (use the counts given).
- Weigh evidence by quality: say when findings come from low-quality studies.
- Do not generalise beyond the included studies. Mention gaps and inconsistencies.
- Do NOT compute a meta-analysis. Write in Markdown with short '###' subheadings, 400-700 words."""

CRITIC_PROMPT = """You are a critical peer reviewer checking a systematic review synthesis
against its study table. Flag: claims not supported by the table, missing or wrong citations,
overgeneralisation, ignoring study quality, and missing discussion of conflicting results.
Approve only if there are no important issues. Keep each issue to one sentence."""


def _study_table(state: ReviewState) -> str:
    q = {x["key"]: x.get("overall", "?") for x in state["quality"]}
    lines = []
    for e in state["extractions"]:
        if not e.get("eligible_confirmed", True):
            continue
        rec = next(s["record"] for s in state["studies"] if s["key"] == e["key"])
        author = (rec.get("authors") or ["Unknown"])[0].split()[-1]
        lines.append(
            f"[{e['key']}] {author} et al. ({rec.get('year')}) | design: {e.get('study_type')} | "
            f"sample: {e.get('sample_or_dataset')} | AI: {e.get('ai_method')} | "
            f"outcomes: {', '.join(e.get('outcomes', []) or [])} | findings: {e.get('key_findings')} | "
            f"direction: {e.get('effect_direction')} | quality: {q.get(e['key'])}")
    return "\n".join(lines)


def _valid_keys(state):
    return {e["key"] for e in state["extractions"] if e.get("eligible_confirmed", True)}


def synthesis_agent(state: ReviewState) -> ReviewState:
    table = _study_table(state)
    counts = Counter(e.get("effect_direction") for e in state["extractions"] if e.get("eligible_confirmed", True))
    user = (f"Review question: {state['protocol']['research_question']}\n\n"
            f"Effect direction counts: {dict(counts)}\n\nSTUDY TABLE:\n{table}")
    feedback = state.get("critic_feedback") or []
    if feedback and not feedback[-1]["approved"]:
        user += ("\n\nYOUR PREVIOUS DRAFT:\n" + state["synthesis"] +
                 "\n\nREVISE IT TO FIX THESE ISSUES:\n- " + "\n- ".join(feedback[-1]["issues"]))

    text = ask_text(get_llm(), SYNTH_PROMPT, user) if table else "_No studies met the inclusion criteria._"
    # Keep the synthesis inside its report section: drop a top title, demote other big headings
    text = re.sub(r"^#{1,2} .*synthesis.*\n+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^#{1,2} ", "### ", text, flags=re.MULTILINE)
    rev = state.get("revisions", -1) + 1
    log(state["run_id"], "synthesis", "draft_written", {"revision": rev, "chars": len(text)})
    save_json(state["run_id"], f"synthesis_v{rev}.json", {"text": text})
    print(f"  synthesis draft v{rev} ({len(text.split())} words)")
    return {"synthesis": text, "revisions": rev}


def critic_agent(state: ReviewState) -> ReviewState:
    text = state["synthesis"]
    # Deterministic check: citations must point to real included studies
    cited = set(re.findall(r"S\d+", " ".join(re.findall(r"\[([^\]]+)\]", text))))
    bad = sorted(cited - _valid_keys(state))
    uncited_paras = [p for p in text.split("\n\n") if len(p.split()) > 40 and not re.search(r"\[S\d+", p)]

    try:
        c = ask_structured(get_llm(json_mode=True), Critique, CRITIC_PROMPT,
                           f"STUDY TABLE:\n{_study_table(state)}\n\nSYNTHESIS:\n{text}")
    except Exception as e:
        c = Critique(approved=True, issues=[f"(critic LLM failed: {e})"])
    issues = list(c.issues)
    if bad:
        issues.append(f"Citations to non-existent studies: {', '.join(bad)}. Remove or correct them.")
    if uncited_paras:
        issues.append(f"{len(uncited_paras)} paragraph(s) contain claims without any [S#] citation.")
    approved = c.approved and not bad and not uncited_paras

    entry = {"revision": state["revisions"], "approved": approved, "issues": issues}
    log(state["run_id"], "critic", "review", entry)
    print(f"  critic: {'APPROVED' if approved else f'{len(issues)} issue(s)'}")
    return {"critic_feedback": (state.get("critic_feedback") or []) + [entry]}


def route_after_critic(state: ReviewState) -> str:
    last = state["critic_feedback"][-1]
    if last["approved"] or state["revisions"] >= CONFIG["workflow"]["max_critic_revisions"]:
        return "report"
    return "synthesis"
