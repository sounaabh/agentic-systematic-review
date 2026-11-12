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
without data, and papers not about the intervention. Give 4-6 criteria of each kind."""


def protocol_agent(state: ReviewState) -> ReviewState:
    protocol = ask_structured(get_llm(json_mode=True), Protocol, SYSTEM, f"Topic: {state['topic']}")
    protocol.year_from = protocol.year_from or CONFIG["search"].get("year_from")

    log(state["run_id"], "protocol", "protocol_generated", protocol.model_dump())
    save_json(state["run_id"], "protocol.json", protocol.model_dump())
    return {"protocol": protocol.model_dump()}
