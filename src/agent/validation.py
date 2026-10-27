"""Semantic checks on LLM decisions, run after pydantic parsing succeeds.

Each validator returns a list of human-readable problems (empty = valid).
The messages double as repair feedback sent back to the model, so they are
short and specific.

Hyperparameter bounds are sized for the bundled datasets (<= 569 rows,
<= 30 features): every allowed configuration cross-validates in seconds on
a laptop, and nothing can request thousands of trees or unbounded boosting.
"""

import pandas as pd

from .metrics import supported_metrics
from .state import AnalysisPlan, FeaturePlan, ModelChoice

CLASSIFICATION_MODELS = (
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
)
REGRESSION_MODELS = (
    "linear_regression",
    "random_forest_regressor",
    "gradient_boosting_regressor",
)

# model -> {param: (min, max, allow_none)}; models not listed accept no params.
HYPERPARAM_BOUNDS = {
    "random_forest_classifier": {"n_estimators": (10, 300, True), "max_depth": (1, 20, True)},
    "random_forest_regressor": {"n_estimators": (10, 300, True), "max_depth": (1, 20, True)},
    "gradient_boosting_classifier": {"n_estimators": (10, 300, True), "max_depth": (1, 5, True)},
    "gradient_boosting_regressor": {"n_estimators": (10, 300, True), "max_depth": (1, 5, True)},
}


def models_for_task(task_type: str) -> tuple[str, ...]:
    return CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS


def validate_metric(metric: str, task_type: str, n_classes) -> list[str]:
    allowed = supported_metrics(task_type, n_classes)
    if metric not in allowed:
        return [f"metric '{metric}' is not supported here; choose one of {allowed}"]
    return []


def validate_plan(plan: AnalysisPlan, task_type: str, n_classes, user_metric) -> list[str]:
    # A user-supplied metric wins, so the planner's suggestion is informational.
    if user_metric is not None:
        return []
    return validate_metric(plan.recommended_metric, task_type, n_classes)


def validate_feature_plan(plan: FeaturePlan, dev_df: pd.DataFrame, target_column: str) -> list[str]:
    errors = []
    features = [c for c in dev_df.columns if c != target_column]
    feature_set = set(features)

    seen: dict[str, str] = {}
    for step in plan.steps:
        if step.column == target_column:
            errors.append(f"'{target_column}' is the target and cannot appear in the feature plan")
        elif step.column not in feature_set:
            errors.append(f"unknown column '{step.column}'")
        elif step.column in seen:
            if seen[step.column] == step.transform:
                errors.append(f"duplicate step for column '{step.column}'")
            else:
                errors.append(
                    f"conflicting steps for column '{step.column}': "
                    f"{seen[step.column]} vs {step.transform}"
                )
        else:
            seen[step.column] = step.transform

    missing = [c for c in features if c not in seen]
    if missing:
        errors.append(f"every feature needs exactly one step; missing: {missing}")

    if seen and all(t == "drop" for t in seen.values()):
        errors.append("plan drops every feature; at least one must remain")

    for col, transform in seen.items():
        if transform in ("scale", "log1p") and not pd.api.types.is_numeric_dtype(dev_df[col]):
            errors.append(f"'{transform}' needs a numeric column; '{col}' is {dev_df[col].dtype}")
        elif transform == "log1p" and dev_df[col].min() < 0:
            errors.append(f"log1p needs non-negative values; '{col}' has min {dev_df[col].min():.4g}")
    return errors


def validate_model_choice(choice: ModelChoice, task_type: str) -> list[str]:
    allowed = models_for_task(task_type)
    if choice.model_name not in allowed:
        return [f"'{choice.model_name}' is not a {task_type} model; choose one of {list(allowed)}"]

    errors = []
    bounds = HYPERPARAM_BOUNDS.get(choice.model_name, {})
    for param in ("n_estimators", "max_depth"):
        value = getattr(choice, param)
        if param not in bounds:
            if value is not None:
                errors.append(f"{choice.model_name} does not take {param}; set it to null")
            continue
        lo, hi, allow_none = bounds[param]
        if value is None:
            if not allow_none:
                errors.append(f"{param} is required for {choice.model_name}")
        elif not lo <= value <= hi:
            errors.append(f"{param}={value} is outside the allowed range [{lo}, {hi}]")
    return errors
