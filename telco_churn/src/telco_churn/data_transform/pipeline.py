# src/data_transform/pipeline.py
import pandas as pd

from .base_transformer import BaseTransformer


class PreprocessingPipeline:
    def __init__(self, steps: list[BaseTransformer]):
        """
        Initialize the pipeline with a sequence of steps.
        Args:
            steps (list[BaseTransformer]): List of transformer instances.
        """
        self.steps = steps

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the pipeline on the data.
        Args:
            data (pd.DataFrame): Input data to preprocess.
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        for step in self.steps:
            data = step.transform(data)
        return data
