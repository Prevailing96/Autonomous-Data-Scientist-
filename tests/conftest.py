"""Shared helpers: scripted decisions and a one-call run harness.

No test needs an API key or network access; every LLM response is scripted.
"""

import pytest

from src.agent.fake_llm import ScriptedLLM
from src.agent.runner import run_pipeline
from src.agent.state import RunConfig
from src.data.datasets import load_dataset


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def features_of(bundle):
    return [c for c in bundle.df.columns if c != bundle.target_column]


def feature_plan(bundle, transform="scale", overrides=None):
    overrides = overrides or {}
    return {
        "reasoning": f"test: {transform}",
        "steps": [{"column": c, "transform": overrides.get(c, transform)} for c in features_of(bundle)],
    }


def model(name="logistic_regression", **params):
    return {"reasoning": "test", "model_name": name, **params}


def critic(action, feedback="test feedback"):
    return {"recommended_action": action, "feedback": feedback}


def plan(metric="roc_auc"):
    return {"approach_summary": "test plan", "recommended_metric": metric}


def run(tmp_path, script, dataset="breast_cancer", narrative="test narrative", **config):
    bundle = load_dataset(dataset)
    llm = ScriptedLLM(script, narrative=narrative)
    cfg = RunConfig(dataset=dataset, **config)
    state = run_pipeline(bundle, cfg, llm, output_root=str(tmp_path))
    return state, llm


def calls_to(llm, node):
    return [p for n, p in llm.calls if n == node]
