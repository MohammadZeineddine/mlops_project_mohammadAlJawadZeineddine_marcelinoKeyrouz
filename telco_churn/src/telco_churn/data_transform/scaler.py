import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .base_transformer import BaseTransformer


class DataScaler(BaseTransformer):
    """
    Scales numerical features using standardization or normalization.
    """

    def __init__(self, scaling_type="standard"):
        """
        Args:
            scaling_type (str): Type of scaling ('standard' or 'minmax').
        """
        if scaling_type not in ["standard", "minmax"]:
            raise ValueError("scaling_type must be 'standard' or 'minmax'")
        self.scaling_type = scaling_type
        self.scaler = StandardScaler() if scaling_type == "standard" else MinMaxScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the scaler to the numerical features.
        Args:
            X (pd.DataFrame): Feature data.
        """
        self.scaler.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the numerical features using the fitted scaler.
        Args:
            X (pd.DataFrame): Feature data to transform.
        Returns:
            pd.DataFrame: Scaled data.
        """
        scaled = self.scaler.transform(X)
        return pd.DataFrame(scaled, columns=X.columns, index=X.index)
