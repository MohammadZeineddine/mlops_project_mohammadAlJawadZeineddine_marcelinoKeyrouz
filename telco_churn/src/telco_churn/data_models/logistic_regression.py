from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .base_model import BaseModel


class LogisticRegression(BaseModel):
    """
    **Logistic Regression Model**

    This class implements the Logistic Regression algorithm for binary classification tasks.
    It inherits from the `BaseModel` class and adheres to its defined interface.

    **Attributes:**

    * `model (SklearnLogisticRegression)`: An instance of the underlying scikit-learn LogisticRegression model.

    **Methods:**

    * **__init__(self, **kwargs):**
        - Initializes the model with optional hyperparameters to be passed to the scikit-learn LogisticRegression constructor.

    * **train(self, X, y):**
        - Trains the model using the provided features (X) and target variable (y).

    * **predict(self, X):**
        - Generates predictions for new data points using the trained model.
        - Takes a DataFrame of features (X) as input and returns an array of predicted class labels.

    * **evaluate(self, X, y):**
        - Evaluates the model's performance on a given dataset.
        - Takes features (X) and ground truth labels (y) as input.
        - Returns a dictionary containing various evaluation metrics
            like accuracy, precision, recall, confusion matrix, and ROC AUC score.
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
            "confusion_matrix": confusion_matrix(y, y_pred),
            "roc_auc": roc_auc_score(y, y_pred),
        }