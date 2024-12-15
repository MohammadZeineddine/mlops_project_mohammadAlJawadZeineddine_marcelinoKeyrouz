from pathlib import Path

import pandas as pd
import pytest

from telco_churn.data_loader.factory import DataLoaderFactory


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("feature1,feature2,target\n1,4,0\n2,5,1\n3,6,0")
    return str(csv_file)


def test_csv_loader(sample_csv: str) -> None:
    loader = DataLoaderFactory.get_loader("csv")
    data = loader.load_data(sample_csv)
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (3, 3)
