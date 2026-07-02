"""v0.2 LangGraph workflow.

Pipeline: test_architect → coder → sandbox_arbiter → prompt_compliance_checker.
All routing is done by writing `next_node` and reading it via conditional edges.
"""

from langgraph.graph import StateGraph, END

from .state import AgenticState
from .nodes import (
    test_architect,
    coder,
    sandbox_arbiter,
    prompt_compliance_checker,
    route_from_sandbox,
    route_from_compliance,
)

workflow = StateGraph(AgenticState)

workflow.add_node("test_architect",              test_architect)
workflow.add_node("coder",                       coder)
workflow.add_node("sandbox_arbiter",             sandbox_arbiter)
workflow.add_node("prompt_compliance_checker",   prompt_compliance_checker)

workflow.set_entry_point("test_architect")

workflow.add_conditional_edges(
    "test_architect",
    lambda state: state["next_node"],
    {"coder": "coder", "test_architect": "test_architect", "FINISH": END},
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
        "test_architect": "test_architect",
        "coder": "coder",
        "prompt_compliance_checker": "prompt_compliance_checker",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "prompt_compliance_checker",
    route_from_compliance,
    {"test_architect": "test_architect", "prompt_compliance_checker": "prompt_compliance_checker", "FINISH": END},
)

app = workflow.compile()
