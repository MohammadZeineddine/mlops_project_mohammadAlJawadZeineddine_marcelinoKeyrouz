from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class BaseModel(ABC, BaseEstimator):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X, y):
        """
        Train the model.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the model.
        Args:
            X (pd.DataFrame): Features.
        Returns:
            np.array: Predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Ground truth.
        Returns:
            dict: Evaluation metrics.
        """
        pass
