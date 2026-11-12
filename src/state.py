"""Data models shared by all agents, and the LangGraph state."""
from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------- Protocol (PICOC: works for clinical, social-science and engineering topics) ----------
class Protocol(BaseModel):
    research_question: str
    population: str = Field(description="Who/what is studied (people, firms, production systems...)")
    intervention: str = Field(description="The technology, method or exposure of interest")
    comparator: str = Field(description="What it is compared with, or 'none / not required'")
    outcomes: list[str] = Field(description="Outcomes measured")
    context: str = Field(description="Setting, e.g. education, startups, manufacturing")
    study_designs: list[str] = Field(description="Eligible designs, e.g. experiment, survey, simulation study")
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    year_from: Optional[int] = None
    languages: list[str] = ["English"]


# ---------- Search strategy ----------
class SearchStrategy(BaseModel):
    concepts: dict[str, list[str]] = Field(
        description="Each key concept mapped to 2-5 synonyms, e.g. {'technology': ['generative AI', 'ChatGPT']}")
    queries: list[str] = Field(
        description="2 to 3 short keyword queries (4-8 words each) combining the concepts")
    rationale: str


# ---------- Records ----------
class Record(BaseModel):
    id: str                       # doi:..., arxiv:..., openalex:..., s2:...
    title: str
    abstract: str = ""
    authors: list[str] = []
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    oa_pdf_url: Optional[str] = None
    sources: list[str] = []


# ---------- Screening ----------
class ScreeningDecision(BaseModel):
    decision: Literal["include", "exclude", "uncertain"]
    criteria_triggered: list[str] = Field(description="Which inclusion/exclusion criteria drove the decision")
    rationale: str = Field(description="One or two sentences citing the title/abstract")
    confidence: float = Field(ge=0, le=1)


class Adjudication(BaseModel):
    decision: Literal["include", "exclude"]
    rationale: str


# ---------- Extraction ----------
class Extraction(BaseModel):
    eligible_confirmed: bool = Field(description="Does the full text confirm the study meets the criteria?")
    exclusion_reason: Optional[str] = None
    study_type: str = Field(description="e.g. randomized experiment, survey, case study, simulation")
    setting: str
    sample_or_dataset: str = Field(description="Participants (n) or dataset/benchmark used")
    ai_method: str = Field(description="AI tool or method studied")
    comparison: str
    outcomes: list[str]
    key_findings: str = Field(description="Main results in 1-3 sentences, with numbers if reported")
    effect_direction: Literal["positive", "negative", "mixed", "no effect", "unclear"]
    limitations: str
    evidence: list[str] = Field(description="Chunk ids (e.g. 'S3-c4') supporting the extracted values")


# ---------- Quality appraisal (simplified, MMAT-inspired) ----------
class QualityAppraisal(BaseModel):
    clear_aims: Literal["yes", "no", "unclear"]
    appropriate_design: Literal["yes", "no", "unclear"]
    adequate_sample_or_data: Literal["yes", "no", "unclear"]
    valid_outcome_measures: Literal["yes", "no", "unclear"]
    limitations_addressed: Literal["yes", "no", "unclear"]
    overall: Literal["high", "moderate", "low"]
    justification: str


# ---------- Critic ----------
class Critique(BaseModel):
    approved: bool
    issues: list[str] = Field(description="Concrete problems: unsupported claims, missing citations, overgeneralization")


# ---------- LangGraph state ----------
class ReviewState(TypedDict, total=False):
    run_id: str
    topic: str
    protocol: dict
    search_strategy: dict
    records: list[dict]              # after deduplication (and cap)
    screening: list[dict]
    included_ids: list[str]
    studies: list[dict]              # included studies with short keys S1, S2...
    extractions: list[dict]
    quality: list[dict]
    synthesis: str
    critic_feedback: list[dict]
    revisions: int
    prisma: dict
    metrics: dict
    report_path: str
