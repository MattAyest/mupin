"""v0.1 LangGraph workflow for the Editing Module.

Pipeline: load_source -> analyze -> plan -> apply -> verify -> FINISH.
All routing is done by writing `next_node` and reading it via conditional edges.
"""

from langgraph.graph import StateGraph, END

from .nodes import (
    analyze,
    apply,
    load_source,
    plan,
    route_from_verify,
    verify,
)
from .state import EditingState

workflow = StateGraph(EditingState)

workflow.add_node("load_source", load_source)
workflow.add_node("analyze", analyze)
workflow.add_node("plan", plan)
workflow.add_node("apply", apply)
workflow.add_node("verify", verify)

workflow.set_entry_point("load_source")

workflow.add_conditional_edges(
    "load_source",
    lambda state: state["next_node"],
    {"analyze": "analyze", "FINISH": END},
)

workflow.add_conditional_edges(
    "analyze",
    lambda state: state["next_node"],
    {"plan": "plan", "analyze": "analyze", "FINISH": END},
)

workflow.add_conditional_edges(
    "plan",
    lambda state: state["next_node"],
    {"apply": "apply", "plan": "plan", "FINISH": END},
)

workflow.add_conditional_edges(
    "apply",
    lambda state: state["next_node"],
    {"verify": "verify", "apply": "apply", "FINISH": END},
)

workflow.add_conditional_edges(
    "verify",
    route_from_verify,
    {
        "plan": "plan",
        "apply": "apply",
        "FINISH": END,
    },
)

app = workflow.compile()
