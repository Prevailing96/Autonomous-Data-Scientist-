"""Bundled sample datasets for the agent to analyze.

All datasets ship inside scikit-learn, so the demo runs with zero
external downloads or credentials beyond the Claude API key.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    target_column: str
    task_type: str  # "classification" or "regression"
    description: str


_REGISTRY = {
    "breast_cancer": (
        load_breast_cancer,
        "classification",
        "Diagnose breast tumors as malignant or benign from digitized "
        "cell nuclei measurements.",
    ),
    "wine": (
        load_wine,
        "classification",
        "Classify the cultivar of origin for a wine sample from its "
        "chemical analysis.",
    ),
    "diabetes": (
        load_diabetes,
        "regression",
        "Predict quantitative disease progression one year after "
        "baseline from patient measurements.",
    ),
}

AVAILABLE_DATASETS = tuple(_REGISTRY.keys())


def load_dataset(name: str) -> DatasetBundle:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(_REGISTRY)}"
        )
    loader_fn, task_type, description = _REGISTRY[name]
    bunch = loader_fn(as_frame=True)
    df = bunch.frame.copy()
    return DatasetBundle(
        name=name,
        df=df,
        target_column="target",
        task_type=task_type,
        description=description,
    )
