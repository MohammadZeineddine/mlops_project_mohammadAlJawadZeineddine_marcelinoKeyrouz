import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .base_transformer import BaseTransformer


class CategoricalEncoder(BaseTransformer):
    """
    **Categorical Feature Encoder**

    This transformer class encodes categorical features into numerical representations
    using either one-hot encoding or label encoding.

    **Attributes:**

    * `encoding_type (str)`: The type of encoding to be applied ('onehot' or 'label').

    **Methods:**

    * **__init__(self, encoding_type="onehot")**
        - Initializes the encoder with the specified encoding type.
        - Raises a ValueError if the provided encoding_type is not 'onehot' or 'label'.

    * **fit(self, X: pd.DataFrame, y: pd.Series = None):**
        - Fits the encoder to the categorical features in the provided DataFrame.
        - Learns the necessary mappings for encoding based on the chosen encoding type.

        Args:
            X (pd.DataFrame): The feature data containing categorical features.
            y (pd.Series, optional): The target data (ignored). Default is None.

    * **transform(self, X: pd.DataFrame) -> pd.DataFrame:**
        - Transforms the categorical features in the DataFrame using the fitted encoder.
        - Applies the learned encoding mappings to convert categorical values to numerical representations.

        Args:
            X (pd.DataFrame): The feature data to be transformed.

        Returns:
            pd.DataFrame: The transformed DataFrame with encoded categorical features.
    """

    def __init__(self, encoding_type="onehot"):
        """
        Args:
            encoding_type (str): Type of encoding ('onehot' or 'label').
        """
        if encoding_type not in ["onehot", "label"]:
            raise ValueError("encoding_type must be 'onehot' or 'label'")
        self.encoding_type = encoding_type
        self.encoder = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fits the encoder to the categorical features.
        Args:
            X (pd.DataFrame): Feature data.
        """
        if self.encoding_type == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder.fit(X)
        elif self.encoding_type == "label":
            self.encoder = {col: LabelEncoder().fit(X[col]) for col in X.columns}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the categorical features using the fitted encoder.
        Args:
            X (pd.DataFrame): Feature data to transform.
        Returns:
            pd.DataFrame: Transformed data.
        """
        if self.encoding_type == "onehot":
            transformed = self.encoder.transform(X)
            return pd.DataFrame(
                transformed,
                columns=self.encoder.get_feature_names_out(X.columns),
                index=X.index,
            )
        elif self.encoding_type == "label":
            transformed = X.copy()
            for col in X.columns:
                transformed[col] = self.encoder[col].transform(X[col])
            return transformed