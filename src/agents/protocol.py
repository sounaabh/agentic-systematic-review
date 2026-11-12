"""Protocol Agent: turns a free-text topic into a structured review protocol (PICOC)."""
from src.audit import log, save_json
from src.config import CONFIG
from src.llm import ask_structured, get_llm
from src.state import Protocol, ReviewState

SYSTEM = """You are an expert systematic reviewer following PRISMA 2020.
Turn the user's topic into a precise review protocol using the PICOC framework
(Population, Intervention, Comparator, Outcomes, Context). The topic may be from
any field (management, education, engineering, medicine).
Inclusion and exclusion criteria must be specific and checkable from a title and
abstract alone. Always exclude: non-English papers, editorials/opinion pieces
without data, and papers not about the intervention. Give 4-6 criteria of each kind.
Do NOT make a control group or comparator mandatory unless the question explicitly
requires one: single-group, pre-post and observational studies are usually eligible.
Do not put a publication year in the criteria (the year range is set separately)."""


def _load_protocol_file(path: str) -> Protocol:
    """User-written protocol (YAML or JSON), e.g. configs/protocol_template.yaml."""
    import yaml
    with open(path, encoding="utf-8") as f:
        return Protocol(**yaml.safe_load(f))   # YAML is a superset of JSON


def protocol_agent(state: ReviewState) -> ReviewState:
    if state.get("protocol_file"):
        protocol = _load_protocol_file(state["protocol_file"])
        log(state["run_id"], "protocol", "protocol_loaded_from_file", {"file": state["protocol_file"]})
    else:
        protocol = ask_structured(get_llm(json_mode=True), Protocol, SYSTEM, f"Topic: {state['topic']}")
    # The year range is a project setting, not an LLM choice (reproducibility)
    protocol.year_from = CONFIG["search"].get("year_from") or protocol.year_from

    log(state["run_id"], "protocol", "protocol_generated", protocol.model_dump())
    save_json(state["run_id"], "protocol.json", protocol.model_dump())
    return {"protocol": protocol.model_dump()}
