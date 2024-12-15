from .logistic_regression import LogisticRegression

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
}


def get_model(name, **kwargs):
    """
    Retrieve a model class by name.
    Args:
        name (str): Model name.
        **kwargs: Additional parameters for model initialization.
    Returns:
        BaseModel: An instance of the requested model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in the registry.")
    return MODEL_REGISTRY[name](**kwargs)
