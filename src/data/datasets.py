"""Bundled sample datasets for the agent to analyze.

All datasets ship inside scikit-learn, so the demo runs with zero
external downloads.
"""

from dataclasses import dataclass, field

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    target_column: str
    task_type: str  # "classification" or "regression"
    description: str
    # Label -> human-readable name, for classification targets.
    class_names: dict = field(default_factory=dict)


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
    class_names = {}
    if task_type == "classification":
        class_names = {i: str(n) for i, n in enumerate(bunch.target_names)}
    return DatasetBundle(
        name=name,
        df=df,
        target_column="target",
        task_type=task_type,
        description=description,
        class_names=class_names,
    )
