import os

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from src.agent import tools as T
from src.agent.errors import DatasetError
from src.agent.state import FeaturePlan, ModelChoice
from src.data.datasets import load_dataset

from .conftest import feature_plan, model


def _dev(name):
    b = load_dataset(name)
    dev, test, info = T.split_dataset(b.df, b.target_column, b.task_type, 0.2, 42)
    return b, dev, test, info


def test_run_eda_writes_figures(tmp_path):
    b, dev, _, _ = _dev("wine")
    summary, figures = T.run_eda(dev, b.target_column, str(tmp_path))
    assert summary["n_rows"] == len(dev)
    assert set(summary["features"]) == set(dev.columns) - {b.target_column}
    assert all(os.path.exists(f) for f in figures)


def test_split_is_stratified_reproducible_and_disjoint():
    b, dev, test, info = _dev("breast_cancer")
    assert info["stratified"] and set(dev.index).isdisjoint(test.index)
    _, dev2, _, _ = _dev("breast_cancer")
    assert list(dev.index) == list(dev2.index)
    full = b.df[b.target_column].mean()
    assert abs(dev[b.target_column].mean() - full) < 0.01
    assert abs(test[b.target_column].mean() - full) < 0.02


def test_pipeline_is_unfitted_until_cv():
    b, dev, _, _ = _dev("wine")
    pipe = T.build_pipeline(FeaturePlan.model_validate(feature_plan(b)), ModelChoice.model_validate(model()), 42)
    with pytest.raises(NotFittedError):
        check_is_fitted(pipe)


def test_scaler_is_fitted_on_training_folds_only():
    b, dev, _, _ = _dev("breast_cancer")
    result = T.evaluate_candidate(
        dev, b.target_column, b.task_type,
        FeaturePlan.model_validate(feature_plan(b, "scale")), ModelChoice.model_validate(model()),
        "roc_auc", 5, 42, return_fold_details=True,
    )
    X = dev.drop(columns=[b.target_column])
    for est, train_idx in zip(result["fold_estimators"], result["fold_indices"]["train"]):
        scaler = est.named_steps["prep"].named_transformers_["scale"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[train_idx].mean().values)
        assert not np.allclose(scaler.mean_, X.mean().values)  # not fit on all dev rows


@pytest.mark.parametrize("metric", ["accuracy", "f1_weighted", "roc_auc"])
def test_primary_metric_controls_the_scorer(metric):
    b, dev, _, _ = _dev("breast_cancer")
    fp, mc = FeaturePlan.model_validate(feature_plan(b)), ModelChoice.model_validate(model())
    result = T.evaluate_candidate(dev, b.target_column, b.task_type, fp, mc, metric, 5, 42)
    X, y = dev.drop(columns=[b.target_column]), dev[b.target_column]
    expected = cross_val_score(T.build_pipeline(fp, mc, 42), X, y, scoring=metric,
                               cv=T.make_cv(y, b.task_type, 5, 42))
    assert result["primary_metric"] == metric
    np.testing.assert_allclose(result["primary"]["folds"], expected)


def test_rmse_reported_positive_and_matches_manual():
    b, dev, _, _ = _dev("diabetes")
    fp, mc = FeaturePlan.model_validate(feature_plan(b, "keep")), ModelChoice.model_validate(model("linear_regression"))
    result = T.evaluate_candidate(dev, b.target_column, b.task_type, fp, mc, "rmse", 5, 42, return_fold_details=True)
    X, y = dev.drop(columns=[b.target_column]), dev[b.target_column]
    manual = [
        np.sqrt(mean_squared_error(y.iloc[te], est.predict(X.iloc[te])))
        for est, te in zip(result["fold_estimators"], result["fold_indices"]["test"])
    ]
    np.testing.assert_allclose(result["primary"]["folds"], manual)
    assert result["primary"]["mean"] > 0
    assert "roc_auc" not in result["metrics"]


def test_roc_auc_unavailable_for_multiclass():
    b, dev, _, _ = _dev("wine")
    with pytest.raises(ValueError, match="not supported"):
        T.evaluate_candidate(dev, b.target_column, b.task_type,
                             FeaturePlan.model_validate(feature_plan(b)), ModelChoice.model_validate(model()),
                             "roc_auc", 5, 42)


def test_strict_log1p_rejects_negative_values():
    with pytest.raises(ValueError, match="negative"):
        T.strict_log1p(np.array([[1.0], [-0.5]]))


def test_infeasible_cv_raises_clear_error():
    y = pd.Series([0] * 40 + [1] * 3)
    with pytest.raises(DatasetError, match="smallest class has 3"):
        T.make_cv(y, "classification", 5, 42)
    with pytest.raises(DatasetError, match="too few"):
        T.make_cv(pd.Series(range(6), dtype=float), "regression", 5, 42)


def test_dataset_with_missing_values_or_text_is_rejected():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"], "target": [0, 1, 0]})
    with pytest.raises(DatasetError, match="non-numeric"):
        T.validate_dataset(df, "target", "classification")
    with pytest.raises(DatasetError, match="missing"):
        T.validate_dataset(df.drop(columns=["b"]), "target", "classification")


def test_final_evaluate_refits_on_all_dev_rows():
    b, dev, test, _ = _dev("breast_cancer")
    out = T.final_evaluate(dev, test, b.target_column, b.task_type,
                           FeaturePlan.model_validate(feature_plan(b)), ModelChoice.model_validate(model()), 42)
    assert out["n_train"] == len(dev) and out["n_test"] == len(test)
    assert set(out["metrics"]) == {"accuracy", "f1_weighted", "roc_auc"}
