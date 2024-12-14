from .minmax_scaler_transfomer import MinMaxScalerTransformer
from .base_transformer import DataTransformer


class TransformerFactory:
    @staticmethod
    def get_transformer(method: str) -> DataTransformer:
        if method == "minmax":
            return MinMaxScalerTransformer()
        raise ValueError(f"Unsupported transformation method: {method}")
