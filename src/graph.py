"""The LangGraph workflow: each agent is a node, ReviewState flows between them.

protocol -> search_strategy -> retrieval -> screening -> extraction
         -> synthesis <-> critic (revision loop) -> report
Human-in-the-loop checkpoints pause the graph after `protocol` and `screening`.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.extraction import extraction_agent
from src.agents.protocol import protocol_agent
from src.agents.screening import screening_agent
from src.agents.search import retrieval_agent, search_strategy_agent
from src.agents.synthesis import critic_agent, route_after_critic, synthesis_agent
from src.report.manuscript import report_agent
from src.state import ReviewState


def build_graph(human_in_the_loop: bool = True):
    g = StateGraph(ReviewState)
    g.add_node("protocol", protocol_agent)
    g.add_node("search_strategy", search_strategy_agent)
    g.add_node("retrieval", retrieval_agent)
    g.add_node("screening", screening_agent)
    g.add_node("extraction", extraction_agent)
    g.add_node("synthesis", synthesis_agent)
    g.add_node("critic", critic_agent)
    g.add_node("report", report_agent)

    g.add_edge(START, "protocol")
    g.add_edge("protocol", "search_strategy")
    g.add_edge("search_strategy", "retrieval")
    g.add_edge("retrieval", "screening")
    g.add_edge("screening", "extraction")
    g.add_edge("extraction", "synthesis")
    g.add_edge("synthesis", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"synthesis": "synthesis", "report": "report"})
    g.add_edge("report", END)

    return g.compile(checkpointer=MemorySaver(),
                     interrupt_after=["protocol", "screening"] if human_in_the_loop else [])
