import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .base_transformer import BaseTransformer


class DataScaler(BaseTransformer):
    """
    **Data Scaler Transformer**

    This transformer class scales numerical features in a DataFrame using
    standardization or min-max normalization. It leverages the scikit-learn
    StandardScaler or MinMaxScaler classes depending on the chosen scaling type.

    **Attributes:**

    * `scaling_type (str)`: The type of scaling to be applied ('standard' or 'minmax').

    **Methods:**

    * **__init__(self, scaling_type="standard")**
        - Initializes the scaler with the specified scaling type.

        Args:
            scaling_type (str): The type of scaling to perform. Defaults to 'standard'.

    * **fit(self, X: pd.DataFrame, y: pd.Series = None):**
        - Fits the scaler to the numerical features in the DataFrame.
        - Learns the statistics (mean and standard deviation for standardization,
            min and max values for normalization) from the data.

        Args:
            X (pd.DataFrame): The feature data containing numerical features.
            y (pd.Series, optional): The target data (ignored). Default is None.

    * **transform(self, X: pd.DataFrame) -> pd.DataFrame:**
        - Transforms the numerical features in the DataFrame using the fitted scaler.
        - Applies the learned scaling transformation to the data.

        Args:
            X (pd.DataFrame): The feature data to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame with scaled numerical features.
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

    def fit(self, X: pd.DataFrame, y: Series = None):
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
