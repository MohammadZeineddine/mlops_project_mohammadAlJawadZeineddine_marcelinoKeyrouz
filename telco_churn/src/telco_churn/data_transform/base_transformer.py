from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """
    **Abstract Base Class for Transformers**

    This abstract class defines a common interface for all transformers
    used within the system. It enforces consistency and facilitates reusability
    of data transformation logic across different components.

    **Methods:**

    * **fit(self, X: pd.DataFrame, y: pd.Series = None):**
        - Fits the transformer to the provided data.
        - This method is typically used to learn parameters or statistics
            from the data that will be used for the transformation process.

        Args:
            X (pd.DataFrame): The feature data to be used for fitting.
            y (pd.Series, optional): The target data (optional). Default is None.

    * **transform(self, X: pd.DataFrame) -> pd.DataFrame:**
        - Transforms the input data using the fitted parameters or knowledge.
        - This method should take a DataFrame containing features and return
            a transformed DataFrame.

        Args:
            X (pd.DataFrame): The feature data to be transformed.

        Returns:
            pd.DataFrame: The transformed data.

    * **fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:**
        - Combines the `fit` and `transform` steps into a single method.
        - This convenience method fits the transformer to the data and then
            immediately transforms the data.

        Args:
            X (pd.DataFrame): The feature data.
            y (pd.Series, optional): The target data (optional). Default is None.

        Returns:
            pd.DataFrame: The transformed data.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Abstract method to be implemented by concrete transformer classes.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to be implemented by concrete transformer classes.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Convenience method for combined fitting and transformation.
        """
        self.fit(X, y)
        return self.transform(X)