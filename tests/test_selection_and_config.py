import pytest
from dotenv import load_dotenv
from pydantic import ValidationError

from src.agent import llm as llm_module
from src.agent.errors import ConfigError
from src.agent.selection import acceptance_status, decide_route, select_best
from src.agent.state import RunConfig


def cand(cid, mean, std=0.01, status="valid"):
    if status != "valid":
        return {"candidate_id": cid, "status": status, "validation": None}
    return {"candidate_id": cid, "status": "valid", "validation": {"primary": {"mean": mean, "std": std}}}


@pytest.mark.parametrize("metric", ["accuracy", "f1_weighted", "roc_auc", "r2"])
def test_higher_is_better_metrics_are_maximized(metric):
    assert select_best([cand("C1", 0.80), cand("C2", 0.90), cand("C3", 0.85)], metric)["candidate_id"] == "C2"


def test_rmse_is_minimized():
    assert select_best([cand("C1", 50.0), cand("C2", 45.0), cand("C3", 60.0)], "rmse")["candidate_id"] == "C2"


def test_tie_breaks_on_lower_std_then_earlier_candidate():
    assert select_best([cand("C1", 0.9, 0.02), cand("C2", 0.9, 0.01)], "accuracy")["candidate_id"] == "C2"
    assert select_best([cand("C1", 0.9, 0.01), cand("C2", 0.9, 0.01)], "accuracy")["candidate_id"] == "C1"


def test_failed_candidates_are_never_selected():
    assert select_best([cand("C1", None, status="failed")], "accuracy") is None
    assert select_best([cand("C1", None, status="failed"), cand("C2", 0.5)], "accuracy")["candidate_id"] == "C2"


def test_route_rules():
    hist = [cand("C1", 0.90)]
    assert decide_route(hist, "accuracy", None, 3, "iterate_model", True) == ("iterate_model", None, None)
    assert decide_route(hist, "accuracy", None, 3, "proceed", True)[:2] == ("stop", "critic_proceeded")
    assert decide_route(hist, "accuracy", 0.85, 3, "iterate_model", True)[:2] == ("stop", "target_met")
    assert decide_route(hist, "accuracy", 0.95, 1, "iterate_model", True)[:2] == ("stop", "budget_exhausted")
    assert decide_route(hist, "accuracy", None, 3, None, True)[:2] == ("stop", "critic_failed")
    # rmse threshold is an upper bound
    assert decide_route([cand("C1", 40.0)], "rmse", 45.0, 3, "iterate_model", True)[1] == "target_met"
    assert decide_route([cand("C1", 50.0)], "rmse", 45.0, 3, "iterate_model", True)[1] is None
    # iterate_model with no feature plan to reuse goes back to feature engineering
    failed = [cand("C1", None, status="failed")]
    route, _, note = decide_route(failed, "accuracy", None, 3, "iterate_model", False)
    assert route == "iterate_features" and note


def test_acceptance_status():
    best = cand("C1", 0.9)
    assert acceptance_status(best, "accuracy", None) == "not_specified"
    assert acceptance_status(best, "accuracy", 0.85) == "met"
    assert acceptance_status(best, "accuracy", 0.95) == "not_met"
    assert acceptance_status(None, "accuracy", 0.5) == "not_met"


def test_run_config_validation():
    with pytest.raises(ValidationError, match="requires an explicit primary_metric"):
        RunConfig(dataset="wine", acceptance_threshold=0.9)
    with pytest.raises(ValidationError, match="between 0 and 1"):
        RunConfig(dataset="wine", primary_metric="accuracy", acceptance_threshold=1.5)
    with pytest.raises(ValidationError):
        RunConfig(dataset="wine", max_candidates=0)
    with pytest.raises(ValidationError):
        RunConfig(dataset="wine", max_repair_attempts=3)


def test_model_name_is_read_after_dotenv_loads(tmp_path, monkeypatch):
    # The module is already imported; the value must still be picked up.
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CLAUDE_MODEL=my-configured-model\n")
    load_dotenv(env_file, override=True)
    try:
        assert llm_module.resolve_model_name() == "my-configured-model"
    finally:
        monkeypatch.delenv("CLAUDE_MODEL", raising=False)


def test_missing_model_name_is_a_clear_error():
    with pytest.raises(ConfigError, match="CLAUDE_MODEL is not set"):
        llm_module.resolve_model_name({})
    with pytest.raises(ConfigError):
        llm_module.resolve_model_name({"CLAUDE_MODEL": "   "})


def test_claude_client_configuration_without_network(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    client = llm_module.ClaudeLLM("my-configured-model")
    assert client.model_name == "my-configured-model"
    assert client._chat.model == "my-configured-model"
    assert client._chat.max_retries == llm_module.SDK_MAX_RETRIES
