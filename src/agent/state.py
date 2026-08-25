"""Shared state schema and structured-output contracts for the agent graph.

The pydantic models below are what each LLM node is forced to return via
LangChain's `with_structured_output`. Keeping decisions structured (rather
than having the LLM write free-form code) means every step is inspectable,
type-checked, and safe to execute deterministically.
"""

from typing import Literal, Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    approach_summary: str = Field(
        description="2-4 sentence plan for how to analyze this dataset."
    )
    primary_metric: str = Field(
        description="The single metric that decides success, e.g. "
        "'roc_auc' or 'rmse'."
    )


class FeatureStep(BaseModel):
    column: str
    transform: Literal["scale", "log1p", "drop", "keep"]


class FeaturePlan(BaseModel):
    reasoning: str
    steps: list[FeatureStep]


class ModelChoice(BaseModel):
    reasoning: str
    model_name: Literal[
        "logistic_regression",
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "linear_regression",
        "random_forest_regressor",
        "gradient_boosting_regressor",
    ]
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None


class Critique(BaseModel):
    satisfied: bool
    feedback: str
    recommended_action: Literal["iterate_features", "iterate_model", "proceed"]


class AgentState(TypedDict, total=False):
    dataset_name: str
    task_type: str
    target_column: str
    description: str
    df: pd.DataFrame

    plan: AnalysisPlan
    eda_summary: dict
    eda_figures: list[str]
    feature_plan: FeaturePlan
    engineered_df: pd.DataFrame
    model_choice: ModelChoice
    metrics: dict
    critique: Critique

    iteration: int
    max_iterations: int
    trace: list[str]
    report_path: str
