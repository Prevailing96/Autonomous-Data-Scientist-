import shutil

from src.agent.state import FeaturePlan, FeatureStep, ModelChoice
from src.agent.tools import apply_feature_plan, run_eda, train_and_evaluate
from src.data.datasets import load_dataset


def test_run_eda(tmp_path):
    bundle = load_dataset("wine")
    summary, figures = run_eda(bundle.df, bundle.target_column, str(tmp_path))
    assert summary["n_rows"] == len(bundle.df)
    assert summary["missing_values"] == 0
    assert len(figures) == 2
    for fig in figures:
        assert (tmp_path / __import__("os").path.basename(fig)).exists()
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_apply_feature_plan_drop_and_scale():
    bundle = load_dataset("wine")
    cols = list(bundle.df.columns)
    plan = FeaturePlan(
        reasoning="test",
        steps=[
            FeatureStep(column=cols[0], transform="drop"),
            FeatureStep(column=cols[1], transform="scale"),
        ],
    )
    engineered = apply_feature_plan(bundle.df, bundle.target_column, plan)
    assert cols[0] not in engineered.columns
    assert abs(engineered[cols[1]].mean()) < 1e-6


def test_train_and_evaluate_classification():
    bundle = load_dataset("breast_cancer")
    choice = ModelChoice(
        reasoning="test", model_name="logistic_regression", n_estimators=None, max_depth=None
    )
    metrics = train_and_evaluate(bundle.df, bundle.target_column, "classification", choice)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "roc_auc" in metrics


def test_train_and_evaluate_regression():
    bundle = load_dataset("diabetes")
    choice = ModelChoice(
        reasoning="test", model_name="linear_regression", n_estimators=None, max_depth=None
    )
    metrics = train_and_evaluate(bundle.df, bundle.target_column, "regression", choice)
    assert "rmse" in metrics
    assert "r2" in metrics
