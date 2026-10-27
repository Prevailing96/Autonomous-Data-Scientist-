import pandas as pd
import pytest

from src.agent.state import AnalysisPlan, FeaturePlan, ModelChoice
from src.agent.validation import validate_feature_plan, validate_model_choice, validate_plan

DEV = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [-1.0, 0.0, 1.0], "c": [5.0, 6.0, 7.0], "target": [0, 1, 0]})


def _plan(*steps):
    return FeaturePlan(reasoning="t", steps=[{"column": c, "transform": t} for c, t in steps])


def _errors(*steps):
    return validate_feature_plan(_plan(*steps), DEV, "target")


def test_complete_plan_is_valid():
    assert _errors(("a", "log1p"), ("b", "scale"), ("c", "drop")) == []


def test_unknown_column():
    assert any("unknown column 'zzz'" in e for e in _errors(("a", "keep"), ("b", "keep"), ("c", "keep"), ("zzz", "keep")))


def test_duplicate_and_conflicting_steps():
    errs = _errors(("a", "keep"), ("a", "keep"), ("b", "keep"), ("b", "drop"), ("c", "keep"))
    assert any("duplicate step for column 'a'" in e for e in errs)
    assert any("conflicting steps for column 'b'" in e for e in errs)


def test_missing_column_violates_one_step_per_feature():
    assert any("missing: ['c']" in e for e in _errors(("a", "keep"), ("b", "keep")))


def test_dropping_every_feature():
    assert any("drops every feature" in e for e in _errors(("a", "drop"), ("b", "drop"), ("c", "drop")))


def test_target_cannot_be_a_feature_step():
    errs = _errors(("a", "keep"), ("b", "keep"), ("c", "keep"), ("target", "drop"))
    assert any("is the target" in e for e in errs)


def test_log1p_on_negative_column_is_rejected():
    assert any("log1p needs non-negative" in e for e in _errors(("a", "keep"), ("b", "log1p"), ("c", "keep")))


def test_scale_on_non_numeric_column_is_rejected():
    dev = DEV.assign(c=["x", "y", "z"])
    errs = validate_feature_plan(_plan(("a", "keep"), ("b", "keep"), ("c", "scale")), dev, "target")
    assert any("needs a numeric column" in e for e in errs)


@pytest.mark.parametrize("task,name,ok", [
    ("classification", "logistic_regression", True),
    ("classification", "linear_regression", False),
    ("regression", "random_forest_classifier", False),
    ("regression", "gradient_boosting_regressor", True),
])
def test_model_family_must_match_task(task, name, ok):
    errs = validate_model_choice(ModelChoice(reasoning="t", model_name=name), task)
    assert (errs == []) == ok


@pytest.mark.parametrize("params,fragment", [
    ({"n_estimators": 5000}, "n_estimators=5000 is outside"),
    ({"n_estimators": 0}, "n_estimators=0 is outside"),
    ({"max_depth": 50}, "max_depth=50 is outside"),
])
def test_hyperparameter_bounds(params, fragment):
    choice = ModelChoice(reasoning="t", model_name="random_forest_classifier", **params)
    assert any(fragment in e for e in validate_model_choice(choice, "classification"))


def test_boosting_depth_is_capped_lower_than_forest():
    choice = ModelChoice(reasoning="t", model_name="gradient_boosting_classifier", max_depth=10)
    assert validate_model_choice(choice, "classification")


def test_linear_models_reject_tree_hyperparameters():
    choice = ModelChoice(reasoning="t", model_name="logistic_regression", n_estimators=100)
    assert any("does not take n_estimators" in e for e in validate_model_choice(choice, "classification"))


def test_planner_metric_must_fit_task_unless_user_overrides():
    bad = AnalysisPlan(approach_summary="t", recommended_metric="roc_auc")
    assert validate_plan(bad, "classification", 3, None)  # roc_auc needs a binary target
    assert validate_plan(bad, "regression", None, None)
    assert validate_plan(bad, "classification", 3, "f1_weighted") == []
