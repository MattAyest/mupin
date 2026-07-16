"""LangGraph state machine for the Planner module.

    clarify → plan → dispatch → inspect → decide
                                                    → dispatch (next step)
                                                    → ASK_OPERATOR (pause)
                                                    → FINISH

When a node returns next_node == "ASK_OPERATOR", the graph pauses — the API
layer stores the state and waits for the operator to POST an answer before
resuming.
"""

from langgraph.graph import StateGraph, END

from .nodes import clarify, decide, dispatch, inspect, plan
from .state import PlannerState

workflow = StateGraph(PlannerState)

workflow.add_node("clarify", clarify)
workflow.add_node("plan", plan)
workflow.add_node("dispatch", dispatch)
workflow.add_node("inspect", inspect)
workflow.add_node("decide", decide)

workflow.set_entry_point("clarify")

workflow.add_conditional_edges(
    "clarify",
    lambda s: s.get("next_node", "FINISH"),
    {
        "plan": "plan",
        "ASK_OPERATOR": END,   # Pause — API stores state, waits for answer
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "plan",
    lambda s: s.get("next_node", "FINISH"),
    {
        "dispatch": "dispatch",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "dispatch",
    lambda s: s.get("next_node", "FINISH"),
    {
        "inspect": "inspect",
        "decide": "decide",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "inspect",
    lambda s: s.get("next_node", "FINISH"),
    {
        "decide": "decide",
        "FINISH": END,
    },
)

workflow.add_conditional_edges(
    "decide",
    lambda s: s.get("next_node", "FINISH"),
    {
        "dispatch": "dispatch",
        "ASK_OPERATOR": END,   # Pause — API stores state, waits for answer
        "FINISH": END,
    },
)

app = workflow.compile()