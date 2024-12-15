# tests/test_data_transform.py
import pandas as pd
import pytest

from telco_churn.data_transform.factory import TransformerFactory


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})


def test_standard_scaler_transform(sample_data: pd.DataFrame) -> None:
    transformer = TransformerFactory.create_transformer("standard")
    transformed_data = transformer.transform(sample_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == sample_data.shape


def test_minmax_scaler_transform(sample_data: pd.DataFrame) -> None:
    transformer = TransformerFactory.create_transformer("minmax")
    transformed_data = transformer.transform(sample_data)
    assert isinstance(transformed_data, pd.DataFrame)
    assert transformed_data.shape == sample_data.shape
