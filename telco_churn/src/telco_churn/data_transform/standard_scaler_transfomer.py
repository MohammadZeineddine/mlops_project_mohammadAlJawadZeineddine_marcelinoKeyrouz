from sklearn.preprocessing import StandardScaler
import pandas as pd
from .base_transformer import DataTransformer


class StandardScalerTransformer(DataTransformer):
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply standard scaling to the data."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)
