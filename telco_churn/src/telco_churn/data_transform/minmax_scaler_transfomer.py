from .base_transformer import DataTransformer
from .minmax_scaler_transfomer import MinMaxScalerTransformer


class TransformerFactory:
    @staticmethod
    def get_transformer(method: str) -> DataTransformer:
        if method == "minmax":
            return MinMaxScalerTransformer()
        raise ValueError(f"Unsupported transformation method: {method}")
