import pytest

from src.data.datasets import AVAILABLE_DATASETS, load_dataset


@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_load_dataset(name):
    bundle = load_dataset(name)
    assert bundle.name == name
    assert bundle.target_column in bundle.df.columns
    assert bundle.task_type in ("classification", "regression")
    assert len(bundle.df) > 0


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")
