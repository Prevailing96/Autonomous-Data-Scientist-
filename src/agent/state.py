"""Run configuration, LLM decision schemas, and the shared graph state.

The pydantic models are the only shapes an LLM decision can take. Parsing
into them is the first check; `validation.py` then checks each decision
against the actual data and task before any code acts on it.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from .metrics import METRICS, MetricName

ModelName = Literal[
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "linear_regression",
    "random_forest_regressor",
    "gradient_boosting_regressor",
]


class RunConfig(BaseModel):
    """User-controlled settings. Nothing in here comes from the LLM."""

    dataset: str
    primary_metric: Optional[MetricName] = Field(
        None, description="Overrides the planner's recommendation when set."
    )
    acceptance_threshold: Optional[float] = Field(
        None, description="Validation score the selected candidate must reach."
    )
    max_candidates: int = Field(
        3, ge=1, le=10, description="Max candidate evaluation attempts, failed ones included."
    )
    max_repair_attempts: int = Field(
        1, ge=0, le=1, description="Re-asks allowed per LLM decision after a bad response."
    )
    seed: int = 42
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(5, ge=2, le=10)

    @model_validator(mode="after")
    def _threshold_needs_metric(self):
        if self.acceptance_threshold is None:
            return self
        if self.primary_metric is None:
            raise ValueError("acceptance_threshold requires an explicit primary_metric")
        spec = METRICS[self.primary_metric]
        t = self.acceptance_threshold
        if spec.name == "rmse" and t < 0:
            raise ValueError("an RMSE threshold must be >= 0")
        if spec.name in ("accuracy", "f1_weighted", "roc_auc") and not 0 <= t <= 1:
            raise ValueError(f"a {spec.name} threshold must be between 0 and 1")
        if spec.name == "r2" and t > 1:
            raise ValueError("an R² threshold must be <= 1")
        return self


class AnalysisPlan(BaseModel):
    approach_summary: str = Field(description="2-4 sentence plan for this dataset.")
    recommended_metric: MetricName = Field(
        description="One metric from the allowed list given in the prompt."
    )


class FeatureStep(BaseModel):
    column: str
    transform: Literal["scale", "log1p", "drop", "keep"]


class FeaturePlan(BaseModel):
    reasoning: str
    steps: list[FeatureStep] = Field(
        description="Exactly one step per input feature column, no more, no less."
    )


class ModelChoice(BaseModel):
    reasoning: str
    model_name: ModelName
    n_estimators: Optional[int] = Field(None, description="Tree ensembles only; null otherwise.")
    max_depth: Optional[int] = Field(None, description="Tree ensembles only; null otherwise.")


class Critique(BaseModel):
    feedback: str
    recommended_action: Literal["iterate_features", "iterate_model", "proceed"]


class AgentState(TypedDict, total=False):
    # Development partition only. The final test partition is never put in
    # graph state; only the final_evaluation node holds a reference to it.
    dev_df: pd.DataFrame
    split_info: dict

    plan: AnalysisPlan
    primary_metric: str
    metric_source: str  # "user" or "planner"
    eda_summary: dict
    eda_figures: list[str]

    current_feature_plan: Optional[FeaturePlan]
    feature_plan_origin: Optional[str]  # candidate whose attempt produced it
    feature_stage_failed: bool
    candidates: Annotated[list[dict], operator.add]
    critiques: Annotated[list[dict], operator.add]
    events: Annotated[list[dict], operator.add]

    next_step: str
    stop_reason: str
    best_candidate_id: Optional[str]
    fatal_error: Optional[dict]
    final_test: Optional[dict]
    narrative: Optional[str]
    outcome: dict
    report_paths: dict
