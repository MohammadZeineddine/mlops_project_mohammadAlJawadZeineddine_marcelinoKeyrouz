# src/data_transform/pipeline.py
import pandas as pd
from .base_transformer import BaseTransformer
from .standard_scaler_transformer import StandardScalerTransformer
from .imputer import Imputer
from .encoder import Encoder


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


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.imputer = Imputer()
        self.encoder = Encoder()
        self.scaler = StandardScalerTransformer()

    def run(self):
        df = pd.read_csv(self.config['input_path'])
        df = self.imputer.fit_transform(df)
        df = self.encoder.fit_transform(df)
        df = self.scaler.fit_transform(df)
        df.to_csv(self.config['output_path'], index=False)
