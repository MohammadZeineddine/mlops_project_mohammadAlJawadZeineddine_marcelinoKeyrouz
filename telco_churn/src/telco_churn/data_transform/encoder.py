from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .base_transformer import BaseTransformer
import pandas as pd


class CategoricalEncoder(BaseTransformer):
    """
    Encodes categorical features using one-hot or label encoding.
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
