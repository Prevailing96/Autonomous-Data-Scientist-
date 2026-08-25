from src.agent.graph import build_graph


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__",
        "plan",
        "eda",
        "feature_engineering",
        "modeling",
        "critic",
        "report",
        "__end__",
    }
    assert expected.issubset(node_names)
