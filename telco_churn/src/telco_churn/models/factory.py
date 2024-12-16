from .registry import get_model


def ModelFactory(name, **kwargs):
    """
    Factory function to create model instances.
    Args:
        name (str): Model name.
        **kwargs: Additional parameters for model initialization.
    Returns:
        BaseModel: An instance of the requested model.
    """
    return get_model(name, **kwargs)
