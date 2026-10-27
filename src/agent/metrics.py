"""Allow-list of evaluation metrics and how each one is scored and compared.

Every metric maps to an explicit scikit-learn scorer string, so the metric
named in the run config is the one actually used in cross-validation and on
the final test partition. RMSE uses sklearn's `neg_root_mean_squared_error`
(sklearn scorers are always "higher is better"); we flip the sign back so
reports show a positive RMSE and candidate comparison minimizes it.
"""

from dataclasses import dataclass
from typing import Literal, Optional

MetricName = Literal["accuracy", "f1_weighted", "roc_auc", "rmse", "r2"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    task_type: str
    scoring: str  # sklearn scorer string passed to cross_validate / get_scorer
    greater_is_better: bool
    negated_scorer: bool = False  # sklearn returns -metric for this scorer
    binary_only: bool = False


METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", "classification", "accuracy", True),
    "f1_weighted": MetricSpec("f1_weighted", "classification", "f1_weighted", True),
    "roc_auc": MetricSpec("roc_auc", "classification", "roc_auc", True, binary_only=True),
    "rmse": MetricSpec(
        "rmse", "regression", "neg_root_mean_squared_error", False, negated_scorer=True
    ),
    "r2": MetricSpec("r2", "regression", "r2", True),
}


def supported_metrics(task_type: str, n_classes: Optional[int] = None) -> list[str]:
    """Metrics valid for this task. ROC AUC is offered only for binary targets."""
    names = []
    for spec in METRICS.values():
        if spec.task_type != task_type:
            continue
        if spec.binary_only and n_classes != 2:
            continue
        names.append(spec.name)
    return names


def from_sklearn_score(spec: MetricSpec, raw: float) -> float:
    """Convert a raw sklearn scorer value to the metric's natural scale."""
    return -raw if spec.negated_scorer else raw


def is_better(a: float, b: float, spec: MetricSpec) -> bool:
    """True if score `a` is strictly better than `b` for this metric."""
    return a > b if spec.greater_is_better else a < b


def meets_threshold(value: float, threshold: float, spec: MetricSpec) -> bool:
    return value >= threshold if spec.greater_is_better else value <= threshold


def direction(spec: MetricSpec) -> str:
    return "maximize" if spec.greater_is_better else "minimize"
