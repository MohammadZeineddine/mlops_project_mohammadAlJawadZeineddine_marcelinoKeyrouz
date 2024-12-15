from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_model(y_true, y_pred):
    """
    Evaluate a model with common metrics.
    """
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }
