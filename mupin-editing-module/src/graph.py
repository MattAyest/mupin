"""v0.2 LangGraph workflow for the Editing Module.

Pipeline with TDD + regression check:

    load_source → analyze → plan → route_on_test_impact
        │ none: apply (code-only)
        │ update_existing/add_new/both: apply_tests → apply_code
        → regression_check (hard gate, max 3 loops)
        → verify (sandbox, max 5 loops)
        → FINISH

All routing is done by writing `next_node` and reading it via conditional edges.
"""

from langgraph.graph import StateGraph, END

from .nodes import (
    analyze,
    apply,
    apply_code,
    apply_tests,
    load_source,
    plan,
    regression_check,
    verify,
)
from .state import EditingState

workflow = StateGraph(EditingState)

workflow.add_node("load_source", load_source)
workflow.add_node("analyze", analyze)
workflow.add_node("plan", plan)
workflow.add_node("apply", apply)
workflow.add_node("apply_tests", apply_tests)
workflow.add_node("apply_code", apply_code)
workflow.add_node("regression_check", regression_check)
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
    {
        "apply": "apply",
        "apply_tests": "apply_tests",
        "plan": "plan",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "apply",
    lambda state: state["next_node"],
    {
        "regression_check": "regression_check",
        "apply": "apply",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "apply_tests",
    lambda state: state["next_node"],
    {
        "apply_code": "apply_code",
        "apply_tests": "apply_tests",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "apply_code",
    lambda state: state["next_node"],
    {
        "regression_check": "regression_check",
        "apply_code": "apply_code",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "regression_check",
    lambda state: state["next_node"],
    {
        "verify": "verify",
        "apply": "apply",
        "apply_code": "apply_code",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "verify",
    lambda state: state["next_node"],
    {
        "apply": "apply",
        "apply_code": "apply_code",
        "apply_tests": "apply_tests",
        "plan": "plan",
        "FINISH": END,
    },
)

app = workflow.compile()