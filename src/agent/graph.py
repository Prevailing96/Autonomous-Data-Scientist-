"""Assembles the LangGraph state machine.

Graph shape:

    START -> plan -> eda -> feature_engineering -> modeling -> critic
                                  ^                     ^        |
                                  |_____iterate_features|        |
                                  |____________iterate_model_____|
                                                                  v
                                                                report -> END

The critic's conditional edge is the reflection loop: it can send the run
back to feature engineering or modeling, or let it proceed to the final
report. `max_iterations` (enforced in critic_node) guarantees termination.
"""

from langgraph.graph import END, START, StateGraph

from . import nodes as N
from .state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", N.plan_node)
    graph.add_node("eda", N.eda_node)
    graph.add_node("feature_engineering", N.feature_engineering_node)
    graph.add_node("modeling", N.modeling_node)
    graph.add_node("critic", N.critic_node)
    graph.add_node("report", N.report_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "eda")
    graph.add_edge("eda", "feature_engineering")
    graph.add_edge("feature_engineering", "modeling")
    graph.add_edge("modeling", "critic")
    graph.add_conditional_edges(
        "critic",
        N.route_after_critic,
        {
            "iterate_features": "feature_engineering",
            "iterate_model": "modeling",
            "proceed": "report",
        },
    )
    graph.add_edge("report", END)

    return graph.compile()
