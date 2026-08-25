"""Deterministic data-science operations invoked by the agent's nodes.

These are plain pandas/scikit-learn functions, parameterized by the
structured decisions the LLM makes (see state.py). Keeping the actual
data manipulation and model training deterministic and code-reviewed -
rather than letting the LLM execute arbitrary generated code - makes the
pipeline safe, reproducible, and easy to explain end to end.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .state import FeaturePlan, ModelChoice


def run_eda(df: pd.DataFrame, target_column: str, output_dir: str) -> tuple[dict, list[str]]:
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "describe": df.describe().round(3).to_dict(),
    }
    if df[target_column].nunique() < 10:
        summary["target_balance"] = df[target_column].value_counts().to_dict()

    figures = []

    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Heatmap")
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
    ax.set_title(f"Distribution of target: {target_column}")
    fig.tight_layout()
    target_path = os.path.join(output_dir, "target_distribution.png")
    fig.savefig(target_path, dpi=120)
    plt.close(fig)
    figures.append(target_path)

    return summary, figures


def apply_feature_plan(
    df: pd.DataFrame, target_column: str, feature_plan: FeaturePlan
) -> pd.DataFrame:
    engineered = df.copy()

    drop_cols = [
        s.column
        for s in feature_plan.steps
        if s.transform == "drop" and s.column in engineered.columns and s.column != target_column
    ]
    engineered = engineered.drop(columns=drop_cols)

    log_cols = [
        s.column
        for s in feature_plan.steps
        if s.transform == "log1p"
        and s.column in engineered.columns
        and s.column != target_column
    ]
    for col in log_cols:
        if (engineered[col] >= 0).all():
            engineered[col] = np.log1p(engineered[col])

    scale_cols = [
        s.column
        for s in feature_plan.steps
        if s.transform == "scale"
        and s.column in engineered.columns
        and s.column != target_column
    ]
    if scale_cols:
        scaler = StandardScaler()
        engineered[scale_cols] = scaler.fit_transform(engineered[scale_cols])

    return engineered


_MODEL_REGISTRY = {
    "logistic_regression": lambda p: LogisticRegression(max_iter=1000),
    "random_forest_classifier": lambda p: RandomForestClassifier(
        n_estimators=p.n_estimators or 100, max_depth=p.max_depth, random_state=42
    ),
    "gradient_boosting_classifier": lambda p: GradientBoostingClassifier(
        n_estimators=p.n_estimators or 100, max_depth=p.max_depth or 3, random_state=42
    ),
    "linear_regression": lambda p: LinearRegression(),
    "random_forest_regressor": lambda p: RandomForestRegressor(
        n_estimators=p.n_estimators or 100, max_depth=p.max_depth, random_state=42
    ),
    "gradient_boosting_regressor": lambda p: GradientBoostingRegressor(
        n_estimators=p.n_estimators or 100, max_depth=p.max_depth or 3, random_state=42
    ),
}


def train_and_evaluate(
    df: pd.DataFrame, target_column: str, task_type: str, model_choice: ModelChoice
) -> dict:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = _MODEL_REGISTRY[model_choice.model_name](model_choice)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {}
    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        metrics["f1"] = float(f1_score(y_test, preds, average="weighted"))
        if hasattr(model, "predict_proba") and y.nunique() == 2:
            proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    else:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["r2"] = float(r2_score(y_test, preds))

    cv_scores = cross_val_score(model, X, y, cv=5)
    metrics["cv_mean"] = float(cv_scores.mean())
    metrics["cv_std"] = float(cv_scores.std())

    return metrics
