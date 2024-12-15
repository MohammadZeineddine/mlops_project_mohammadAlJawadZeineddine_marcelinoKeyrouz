from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


class LogisticRegression(BaseModel):
    """
    Logistic Regression model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Logistic Regression model with optional hyperparameters.
        """
        self.model = SklearnLogisticRegression(**kwargs)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
        }
