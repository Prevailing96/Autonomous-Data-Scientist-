"""Assembles the LangGraph state machine.

    START -> plan -> eda -> feature_engineering -> modeling -> critic
                                                                 |
             critic routes:  iterate_features -> feature_engineering
                             iterate_model    -> modeling
                             stop             -> final_evaluation -> report -> END

Shortcuts for failures: plan/eda failures go straight to report, and a
feature_engineering failure goes straight to critic (the attempt is recorded
as a failed candidate). Each pass through feature_engineering/modeling is one
candidate attempt. critic_node applies the code-enforced stopping rules
(selection.decide_route) to the critic's recommendation. final_evaluation is
the only node that receives the test partition.
"""

from functools import partial

import pandas as pd
from langgraph.graph import END, START, StateGraph

from . import nodes as N
from .state import AgentState


def build_graph(deps: N.Deps, holdout: pd.DataFrame):
    graph = StateGraph(AgentState)

    graph.add_node("plan", partial(N.plan_node, deps=deps))
    graph.add_node("eda", partial(N.eda_node, deps=deps))
    graph.add_node("feature_engineering", partial(N.feature_engineering_node, deps=deps))
    graph.add_node("modeling", partial(N.modeling_node, deps=deps))
    graph.add_node("critic", partial(N.critic_node, deps=deps))
    graph.add_node("final_evaluation", partial(N.final_evaluation_node, deps=deps, holdout=holdout))
    graph.add_node("report", partial(N.report_node, deps=deps))

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", N.route_if_fatal, {"continue": "eda", "report": "report"})
    graph.add_conditional_edges(
        "eda", N.route_if_fatal, {"continue": "feature_engineering", "report": "report"}
    )
    graph.add_conditional_edges(
        "feature_engineering", N.route_after_features, {"modeling": "modeling", "critic": "critic"}
    )
    graph.add_edge("modeling", "critic")
    graph.add_conditional_edges(
        "critic",
        N.route_after_critic,
        {
            "iterate_features": "feature_engineering",
            "iterate_model": "modeling",
            "stop": "final_evaluation",
        },
    )
    graph.add_edge("final_evaluation", "report")
    graph.add_edge("report", END)

    return graph.compile()


def recursion_limit(max_candidates: int) -> int:
    """Upper bound on node executions: setup + 3 per attempt + final steps, with slack."""
    return 10 + 3 * max_candidates
