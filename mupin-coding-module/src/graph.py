"""v0.2 LangGraph workflow.

Pipeline: test_designer → skeleton_maker → coder → sandbox_arbiter → FINISH.
All routing is done by writing `next_node` and reading it via conditional edges.
"""

from langgraph.graph import StateGraph, END

from .state import AgenticState
from .nodes import (
    test_designer,
    skeleton_maker,
    coder,
    sandbox_arbiter,
    route_from_sandbox,
)

workflow = StateGraph(AgenticState)

workflow.add_node("test_designer",               test_designer)
workflow.add_node("skeleton_maker",              skeleton_maker)
workflow.add_node("coder",                       coder)
workflow.add_node("sandbox_arbiter",             sandbox_arbiter)

workflow.set_entry_point("test_designer")

workflow.add_conditional_edges(
    "test_designer",
    lambda state: state["next_node"],
    {"skeleton_maker": "skeleton_maker", "test_designer": "test_designer", "FINISH": END},
)

workflow.add_conditional_edges(
    "skeleton_maker",
    lambda state: state["next_node"],
    {"coder": "coder", "test_designer": "test_designer", "FINISH": END},
)

workflow.add_conditional_edges(
    "coder",
    lambda state: state["next_node"],
    {"sandbox_arbiter": "sandbox_arbiter", "coder": "coder", "FINISH": END},
)

workflow.add_conditional_edges(
    "sandbox_arbiter",
    route_from_sandbox,
    {
        "test_designer": "test_designer",
        "coder": "coder",
        "FINISH": END,
    },
)

app = workflow.compile()
