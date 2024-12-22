from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base_model import BaseModel


class RandomForest(BaseModel):
    """
    Random Forest model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model with optional hyperparameters.
        """
        self.model = SklearnRandomForestClassifier(**kwargs)

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
            "confusion_matrix": confusion_matrix(y, y_pred),
            "roc_auc": roc_auc_score(y, y_pred),
        }
