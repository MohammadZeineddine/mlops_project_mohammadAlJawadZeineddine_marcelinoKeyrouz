from .encoder import CategoricalEncoder
from .imputer import DataImputer
from .scaler import DataScaler


class TransformerFactory:
    """
    **Transformer Factory**

    This class provides a factory method for creating instances of different
    transformer classes. This centralizes the creation process and makes it
    easier to manage and extend the available transformers.

    **Methods:**

    * **get_transformer(name: str, **kwargs)**
        - Creates an instance of the specified transformer.

        Args:
            name (str):
                The name of the transformer to be created ('encoder', 'imputer', 'scaler').

            **kwargs:
                Additional keyword arguments to be passed to the transformer's constructor.

        Returns:
            BaseTransformer:
                An instance of the requested transformer class.

        Raises:
            ValueError:
                If the provided `name` does not correspond to any known transformer.
    """

    @staticmethod
    def get_transformer(name: str, **kwargs):
        """
        Creates an instance of the specified transformer.
        """
        if name == "encoder":
            return CategoricalEncoder(**kwargs)
        elif name == "imputer":
            return DataImputer(**kwargs)
        elif name == "scaler":
            return DataScaler(**kwargs)
        else:
            raise ValueError(f"Unknown transformer name: {name}")