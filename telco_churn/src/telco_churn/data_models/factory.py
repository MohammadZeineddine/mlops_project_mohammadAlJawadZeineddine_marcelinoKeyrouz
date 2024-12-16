from .random_forest import RandomForest
from .logistic_regression import LogisticRegression

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForest,
}


def ModelFactory(name, **kwargs):
    """
    Factory function to create model instances.
    Args:
        name (str): Model name.
        **kwargs: Additional parameters for model initialization.
    Returns:
        BaseModel: An instance of the requested model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in the registry.")
    return MODEL_REGISTRY[name](**kwargs)
