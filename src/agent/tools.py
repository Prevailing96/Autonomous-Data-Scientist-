"""Deterministic data-science operations invoked by the agent's nodes.

Plain pandas/scikit-learn functions, parameterized by the structured
decisions the LLM makes. Feature plans become an *unfitted* sklearn
Pipeline, so every fit (scaling included) happens inside a training fold
during cross-validation, and once on the full development set at the end.
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .errors import DatasetError
from .metrics import METRICS, from_sklearn_score, supported_metrics
from .state import FeaturePlan, ModelChoice


# --------------------------------------------------------------------------
# Dataset checks and partitioning
# --------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame, target_column: str, task_type: str) -> None:
    """Reject data this pipeline cannot handle, instead of guessing.

    There is no imputer or categorical encoder in the allowed transforms, so
    missing values and non-numeric features are rejected up front.
    """
    if target_column not in df.columns:
        raise DatasetError(f"target column '{target_column}' not found")
    features = df.drop(columns=[target_column])
    if features.shape[1] == 0:
        raise DatasetError("dataset has no feature columns")
    non_numeric = [c for c in features.columns if not pd.api.types.is_numeric_dtype(features[c])]
    if non_numeric:
        raise DatasetError(f"non-numeric feature columns are not supported: {non_numeric}")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        raise DatasetError(f"dataset has {n_missing} missing values; no imputation is configured")
    if task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_column]):
        raise DatasetError("regression target must be numeric")


def split_dataset(
    df: pd.DataFrame, target_column: str, task_type: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split once into development and final test partitions."""
    stratify = None
    if task_type == "classification":
        counts = df[target_column].value_counts()
        if counts.min() < 2:
            raise DatasetError(
                f"cannot stratify: class {counts.idxmin()!r} has only {counts.min()} row(s)"
            )
        stratify = df[target_column]
    dev, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
    info = {
        "seed": seed,
        "test_size": test_size,
        "stratified": stratify is not None,
        "n_dev": len(dev),
        "n_test": len(test),
    }
    if task_type == "classification":
        info["dev_class_counts"] = {str(k): int(v) for k, v in dev[target_column].value_counts().sort_index().items()}
        info["test_class_counts"] = {str(k): int(v) for k, v in test[target_column].value_counts().sort_index().items()}
    return dev, test, info


def make_cv(y: pd.Series, task_type: str, n_folds: int, seed: int):
    """Build the CV splitter, failing loudly if the folds would be degenerate."""
    if len(y) < 2 * n_folds:
        raise DatasetError(f"{len(y)} development rows is too few for {n_folds}-fold CV")
    if task_type == "classification":
        smallest = int(y.value_counts().min())
        if smallest < n_folds:
            raise DatasetError(
                f"smallest class has {smallest} development rows; "
                f"stratified {n_folds}-fold CV needs at least {n_folds}"
            )
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return KFold(n_splits=n_folds, shuffle=True, random_state=seed)


# --------------------------------------------------------------------------
# EDA (development partition only)
# --------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, target_column: str, output_dir: str) -> tuple[dict, list[str]]:
    os.makedirs(output_dir, exist_ok=True)
    features = df.drop(columns=[target_column])

    summary = {
        "n_rows": len(df),
        "n_features": features.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        # Aggregates only - no raw rows go into prompts or the trace.
        "features": {
            c: {
                "dtype": str(features[c].dtype),
                "min": round(float(features[c].min()), 4),
                "max": round(float(features[c].max()), 4),
                "mean": round(float(features[c].mean()), 4),
                "skew": round(float(features[c].skew()), 3),
            }
            for c in features.columns
        },
    }
    if df[target_column].nunique() < 10:
        summary["target_balance"] = {str(k): int(v) for k, v in df[target_column].value_counts().sort_index().items()}
    else:
        summary["target_stats"] = {
            "min": round(float(df[target_column].min()), 3),
            "max": round(float(df[target_column].max()), 3),
            "mean": round(float(df[target_column].mean()), 3),
        }

    figures = []

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature correlation (development partition)")
    fig.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(corr_path, dpi=120)
    plt.close(fig)
    figures.append(corr_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    if df[target_column].nunique() < 10:
        df[target_column].value_counts().sort_index().plot(kind="bar", ax=ax)
    else:
        ax.hist(df[target_column], bins=30)
    ax.set_title(f"Target distribution: {target_column} (development partition)")
    fig.tight_layout()
    target_path = os.path.join(output_dir, "target_distribution.png")
    fig.savefig(target_path, dpi=120)
    plt.close(fig)
    figures.append(target_path)

    return summary, figures


# --------------------------------------------------------------------------
# Pipeline construction (unfitted)
# --------------------------------------------------------------------------

def strict_log1p(X):
    """log1p that refuses negative inputs instead of silently producing NaN."""
    arr = np.asarray(X, dtype=float)
    if (arr < 0).any():
        raise ValueError("log1p transform received negative values")
    return np.log1p(arr)


def build_preprocessor(feature_plan: FeaturePlan) -> ColumnTransformer:
    by_transform: dict[str, list[str]] = {"scale": [], "log1p": [], "keep": []}
    for step in feature_plan.steps:
        if step.transform != "drop":
            by_transform[step.transform].append(step.column)

    transformers = []
    if by_transform["scale"]:
        transformers.append(("scale", StandardScaler(), by_transform["scale"]))
    if by_transform["log1p"]:
        transformers.append(
            ("log1p", FunctionTransformer(strict_log1p, feature_names_out="one-to-one"), by_transform["log1p"])
        )
    if by_transform["keep"]:
        transformers.append(("keep", "passthrough", by_transform["keep"]))
    # Columns marked "drop" are the only ones left for `remainder`.
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(choice: ModelChoice, seed: int):
    n, d = choice.n_estimators, choice.max_depth
    registry = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "random_forest_classifier": lambda: RandomForestClassifier(
            n_estimators=n or 100, max_depth=d, random_state=seed
        ),
        "gradient_boosting_classifier": lambda: GradientBoostingClassifier(
            n_estimators=n or 100, max_depth=d or 3, random_state=seed
        ),
        "linear_regression": lambda: LinearRegression(),
        "random_forest_regressor": lambda: RandomForestRegressor(
            n_estimators=n or 100, max_depth=d, random_state=seed
        ),
        "gradient_boosting_regressor": lambda: GradientBoostingRegressor(
            n_estimators=n or 100, max_depth=d or 3, random_state=seed
        ),
    }
    return registry[choice.model_name]()


def build_pipeline(feature_plan: FeaturePlan, choice: ModelChoice, seed: int) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(feature_plan)), ("model", build_model(choice, seed))])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _metrics_for(task_type: str, y: pd.Series) -> list[str]:
    n_classes = int(y.nunique()) if task_type == "classification" else None
    return supported_metrics(task_type, n_classes)


def evaluate_candidate(
    dev_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    feature_plan: FeaturePlan,
    choice: ModelChoice,
    primary_metric: str,
    cv_folds: int,
    seed: int,
    return_fold_details: bool = False,
) -> dict:
    """Cross-validate an unfitted pipeline on development data only.

    Preprocessing is part of the pipeline, so sklearn fits it on each
    training fold and applies it to the matching validation fold.
    """
    X = dev_df.drop(columns=[target_column])
    y = dev_df[target_column]
    metric_names = _metrics_for(task_type, y)
    if primary_metric not in metric_names:
        raise ValueError(f"metric '{primary_metric}' is not supported for this data")

    scoring = {m: METRICS[m].scoring for m in metric_names}
    cv = make_cv(y, task_type, cv_folds, seed)
    pipeline = build_pipeline(feature_plan, choice, seed)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score="raise",  # a failed fold is an error, not a NaN score
            return_estimator=return_fold_details,
            return_indices=return_fold_details,
        )

    metrics = {}
    for m in metric_names:
        folds = [from_sklearn_score(METRICS[m], float(s)) for s in result[f"test_{m}"]]
        metrics[m] = {"mean": float(np.mean(folds)), "std": float(np.std(folds)), "folds": folds}

    out = {
        "primary_metric": primary_metric,
        "primary": metrics[primary_metric],
        "metrics": metrics,
        "n_folds": cv_folds,
        "warnings": sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught}),
    }
    if return_fold_details:
        out["fold_estimators"] = result["estimator"]
        out["fold_indices"] = result["indices"]
    return out


def final_evaluate(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    feature_plan: FeaturePlan,
    choice: ModelChoice,
    seed: int,
) -> dict:
    """Fit a fresh pipeline on all development data; score it once on the test partition."""
    pipeline = build_pipeline(feature_plan, choice, seed)  # new, unfitted instance
    X_dev, y_dev = dev_df.drop(columns=[target_column]), dev_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_dev, y_dev)
    metrics = {}
    for m in _metrics_for(task_type, y_dev):
        spec = METRICS[m]
        metrics[m] = from_sklearn_score(spec, float(get_scorer(spec.scoring)(pipeline, X_test, y_test)))
    return {"metrics": metrics, "n_train": len(dev_df), "n_test": len(test_df)}
