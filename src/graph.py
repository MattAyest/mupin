from langgraph.graph import StateGraph, END

from .state import SwarmState
from .nodes import (
    workspace_loader,
    architect_node,
    test_writer,
    contract_verifier,
    code_writer,
    static_analyzer,
    deterministic_verifier,
    error_distiller,
    archivist_node,
)

workflow = StateGraph(SwarmState)

workflow.add_node("workspace_loader",        workspace_loader)
workflow.add_node("architect_node",          architect_node)
workflow.add_node("test_writer",             test_writer)
workflow.add_node("contract_verifier",       contract_verifier)
workflow.add_node("code_writer",             code_writer)
workflow.add_node("static_analyzer",         static_analyzer)
workflow.add_node("deterministic_verifier",  deterministic_verifier)
workflow.add_node("error_distiller",         error_distiller)
workflow.add_node("archivist_node",          archivist_node)

workflow.set_entry_point("workspace_loader")

workflow.add_conditional_edges("workspace_loader",       lambda x: x["next_node"], {"architect_node":         "architect_node"})
workflow.add_conditional_edges("architect_node",         lambda x: x["next_node"], {"test_writer":            "test_writer"})
workflow.add_conditional_edges("test_writer",            lambda x: x["next_node"], {"contract_verifier":      "contract_verifier",   "error_distiller": "error_distiller"})
workflow.add_conditional_edges("contract_verifier",      lambda x: x["next_node"], {"code_writer":            "code_writer",         "test_writer":     "test_writer",     "static_analyzer": "static_analyzer"})
workflow.add_conditional_edges("code_writer",            lambda x: x["next_node"], {"static_analyzer":        "static_analyzer",     "error_distiller": "error_distiller"})
workflow.add_conditional_edges("static_analyzer",        lambda x: x["next_node"], {"deterministic_verifier": "deterministic_verifier", "error_distiller": "error_distiller"})
workflow.add_conditional_edges("deterministic_verifier", lambda x: x["next_node"], {"error_distiller":  "error_distiller",  "FINISH": END})
workflow.add_conditional_edges("error_distiller",        lambda x: x["next_node"], {"code_writer":      "code_writer",      "test_writer": "test_writer", "architect_node": "architect_node", "archivist_node": "archivist_node", "FINISH": END})
workflow.add_conditional_edges("archivist_node",         lambda x: x["next_node"], {"FINISH": END})

app = workflow.compile()
