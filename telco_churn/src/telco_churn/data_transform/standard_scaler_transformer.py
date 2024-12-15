from sklearn.preprocessing import StandardScaler
import pandas as pd


class StandardScalerTransformer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame):
        self.scaler.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
