from .logistic_regression import LogisticRegression
from .random_forest import RandomForest

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForest,
}


def ModelFactory(name, **kwargs):
    """
    **Creates instances of models registered in the registry.**

    This factory function provides a convenient way to instantiate different
    model classes based on their names.

    Args:
        name (str):
            The name of the desired model.
            Must be a key in the `MODEL_REGISTRY`.

        **kwargs:
            Additional keyword arguments to be passed to the model constructor.

    Returns:
        BaseModel:
            An instance of the requested model class.

    Raises:
        ValueError:
            If the specified `name` is not found in the `MODEL_REGISTRY`.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found in the registry.")
    return MODEL_REGISTRY[name](**kwargs)