from __future__ import annotations
import pandas as pd
from loguru import logger
import joblib

from telco_churn.data_transform.factory import TransformerFactory
from telco_churn.data_models.factory import ModelFactory


def load_pipeline(preprocessing_path: str, model_path: str, feature_names_path: str) -> InferencePipeline:
    """
    Load the saved preprocessing pipeline, trained model, and feature names.

    Args:
        preprocessing_path (str): Path to the saved preprocessing pipeline.
        model_path (str): Path to the saved trained model.
        feature_names_path (str): Path to the saved feature names.

    Returns:
        InferencePipeline: Initialized inference pipeline.
    """
    logger.info(
        "Loading preprocessing pipeline, trained model, and feature names.")
    preprocessing_pipeline = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    with open(feature_names_path, "r") as f:
        feature_names = f.read().splitlines()
    logger.success(
        "Preprocessing pipeline, model, and feature names loaded successfully.")
    return InferencePipeline(preprocessing_pipeline, model, feature_names)


class InferencePipeline:
    """
    Class to handle the entire inference pipeline, including preprocessing and model prediction.
    """
    _preprocessing_pipeline: list
    _model: object
    _feature_names: list

    def __init__(self, preprocessing_pipeline: list, model: object, feature_names: list):
        """
        Initialize the inference pipeline.

        Args:
            preprocessing_pipeline (list): The saved preprocessing pipeline steps.
            model (object): The trained machine learning model.
            feature_names (list): Feature names seen during training.
        """
        self._preprocessing_pipeline = preprocessing_pipeline
        self._model = model
        self._feature_names = feature_names

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the preprocessing pipeline and model prediction on input data.

        Args:
            data (pd.DataFrame): The input raw data for inference.

        Returns:
            pd.DataFrame: Predictions as a DataFrame.
        """
        try:
            logger.info("Pipeline execution started.")

            # Apply the preprocessing pipeline
            logger.info("Applying data transformation.")
            transformed_data = self._apply_pipeline(data)
            logger.debug(f"Transformed Data:\n{transformed_data.head()}")
            logger.success("Data transformed successfully.")

            # Align feature names
            logger.info(
                "Aligning features to match the model's input requirements.")
            aligned_data = self._align_features(transformed_data)
            logger.debug(f"Aligned Data:\n{aligned_data.head()}")

            # Make predictions
            logger.info("Running model inference.")
            predictions = self._model.predict(aligned_data)
            logger.debug(f"Predictions: {predictions}")
            logger.success("Model inference completed successfully.")

            # Return predictions as a DataFrame
            prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
            logger.info("Pipeline execution completed successfully.")
            return prediction_df

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

    def _apply_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline step-by-step to the input data.

        Args:
            data (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Transformed data ready for model inference.
        """
        for step_name, transformer, columns in self._preprocessing_pipeline:
            logger.debug(f"Applying step '{step_name}' on columns {columns}.")
            transformed_part = transformer.transform(data[columns])

            # Handle multi-column output transformers
            if hasattr(transformer, "get_feature_names_out"):
                feature_names = transformer.get_feature_names_out(columns)
            else:
                feature_names = columns

            transformed_df = pd.DataFrame(
                transformed_part, columns=feature_names, index=data.index)
            data = data.drop(columns=columns).reset_index(drop=True)
            transformed_df = transformed_df.reset_index(drop=True)
            data = pd.concat([data, transformed_df], axis=1)
        return data

    def _align_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align transformed data's features to match those seen during training.

        Args:
            data (pd.DataFrame): Transformed input data.

        Returns:
            pd.DataFrame: Data aligned with the model's expected feature names.
        """
        aligned_data = data.reindex(columns=self._feature_names, fill_value=0)
        return aligned_data
