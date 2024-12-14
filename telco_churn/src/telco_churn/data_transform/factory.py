from typing import Any, Dict
from .imputer import MissingValueImputer
from .encoder import CategoricalEncoder
from .scaler import NumericalScaler
from .pipeline import PreprocessingPipeline
from .base_transformer import BaseTransformer


class TransformerFactory:
    @staticmethod
    def create_transformer(name: str, params: Dict[str, Any]) -> BaseTransformer:
        """
        Factory method to create a transformer instance.
        Args:
            name (str): Name of the transformer.
            params (dict): Parameters for the transformer.
        Returns:
            BaseTransformer: An instance of the requested transformer.
        """
        if name == "imputer":
            return MissingValueImputer(**params)
        elif name == "encoder":
            return CategoricalEncoder(**params)
        elif name == "scaler":
            return NumericalScaler(**params)
        else:
            raise ValueError(f"Unsupported transformer name: {name}")


class PreprocessingPipelineFactory:
    @staticmethod
    def create_pipeline(config: Dict[str, Any]) -> PreprocessingPipeline:
        """
        Create a preprocessing pipeline from a configuration.
        Args:
            config (dict): Configuration dictionary with pipeline steps.
        Returns:
            PreprocessingPipeline: A fully constructed preprocessing pipeline.
        """
        steps = []
        for step in config["pipeline"]:
            transformer_name = step["name"]
            transformer_params = step.get("params", {})
            transformer = TransformerFactory.create_transformer(
                transformer_name, transformer_params)
            steps.append(transformer)

        return PreprocessingPipeline(steps=steps)
