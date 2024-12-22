from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class BaseModel(ABC, BaseEstimator):
    """
    **Abstract Base Class for All Data Models**

    This class serves as a foundational blueprint for all data models within the system.
    It defines essential methods that every model must implement:

    * **train(X, y):**
        Trains the model using the provided features (X) and target variable (y).
        This method must be implemented by each concrete model class.

    * **predict(X):**
        Generates predictions for new data using the trained model.
        The input should be a DataFrame containing features.
        The output should be an array of predictions.

    * **evaluate(X, y):**
        Assesses the model's performance on a given dataset.
        Takes features (X) and ground truth labels (y) as input.
        Returns a dictionary containing relevant evaluation metrics
        (e.g., accuracy, precision, recall, F1-score).

    This abstract class ensures that all models adhere to a common interface,
    facilitating consistency and interoperability within the system.
    """

    @abstractmethod
    def train(self, X, y):
        """
        **Trains the model.**

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        **Makes predictions using the model.**

        Args:
            X (pd.DataFrame): Features.

        Returns:
            np.array: Predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        **Evaluates the model on test data.**

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Ground truth.

        Returns:
            dict: Evaluation metrics.
        """
        pass