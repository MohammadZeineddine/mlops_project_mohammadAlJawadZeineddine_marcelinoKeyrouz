from .encoder import CategoricalEncoder
from .imputer import DataImputer
from .scaler import DataScaler


def transformer_factory(name: str, **kwargs):
    """
    Factory function to create transformer instances.

    Args:
        name (str): Name of the transformer ('encoder', 'imputer', 'scaler').
        **kwargs: Additional arguments for the transformer class.

    Returns:
        BaseTransformer: Instance of the requested transformer.
    """
    if name == "encoder":
        return CategoricalEncoder(**kwargs)
    elif name == "imputer":
        return DataImputer(**kwargs)
    elif name == "scaler":
        return DataScaler(**kwargs)
    else:
        raise ValueError(f"Unknown transformer name: {name}")
