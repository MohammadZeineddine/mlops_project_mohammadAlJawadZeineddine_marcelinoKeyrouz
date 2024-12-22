from __future__ import annotations

import argparse

import joblib
import pandas as pd
from loguru import logger


def load_pipeline(
        preprocessing_path: str, model_path: str, feature_names_path: str
) -> InferencePipeline:
    """
    Load the saved preprocessing pipeline, trained model, and feature names.
    """
    logger.info("Loading preprocessing pipeline, trained model, and feature names.")
    preprocessing_pipeline = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    with open(feature_names_path, "r") as f:
        feature_names = f.read().splitlines()
    logger.success(
        "Preprocessing pipeline, model, and feature names loaded successfully."
    )
    return InferencePipeline(preprocessing_pipeline, model, feature_names)


class InferencePipeline:
    """
    Class to handle the entire inference pipeline, including preprocessing and model prediction.
    """

    def __init__(
            self, preprocessing_pipeline: list, model: object, feature_names: list
    ):
        self._preprocessing_pipeline = preprocessing_pipeline
        self._model = model
        self._feature_names = feature_names

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run the preprocessing pipeline and model prediction on input data.
        """
        try:
            logger.info("Pipeline execution started.")

            logger.info("Applying data transformation.")
            transformed_data = self._apply_pipeline(data)
            logger.debug(f"Transformed Data:\n{transformed_data.head()}")
            logger.success("Data transformed successfully.")

            logger.info("Aligning features to match the model's input requirements.")
            aligned_data = self._align_features(transformed_data)
            logger.debug(f"Aligned Data:\n{aligned_data.head()}")

            logger.info("Running model inference.")
            predictions = self._model.predict(aligned_data)
            logger.debug(f"Predictions: {predictions}")
            logger.success("Model inference completed successfully.")

            prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
            logger.info("Pipeline execution completed successfully.")
            return prediction_df

        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}")
            raise

    def _apply_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the preprocessing pipeline step-by-step to the input data.
        """
        for step_name, transformer, columns in self._preprocessing_pipeline:
            logger.debug(f"Applying step '{step_name}' on columns {columns}.")
            transformed_part = transformer.transform(data[columns])

            if hasattr(transformer, "get_feature_names_out"):
                feature_names = transformer.get_feature_names_out(columns)
            else:
                feature_names = columns

            transformed_df = pd.DataFrame(
                transformed_part, columns=feature_names, index=data.index
            )
            data = data.drop(columns=columns).reset_index(drop=True)
            transformed_df = transformed_df.reset_index(drop=True)
            data = pd.concat([data, transformed_df], axis=1)
        return data

    def _align_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align transformed data's features to match those seen during training.
        """
        aligned_data = data.reindex(columns=self._feature_names, fill_value=0)
        return aligned_data


def main():
    """
    Main entry point for running the pipeline directly.
    """
    parser = argparse.ArgumentParser(description="Run inference pipeline directly.")
    parser.add_argument(
        "--preprocessing", required=True, help="Path to preprocessing pipeline file."
    )
    parser.add_argument("--model", required=True, help="Path to trained model file.")
    parser.add_argument("--features", required=True, help="Path to feature names file.")
    parser.add_argument("--data", required=True, help="Path to input data CSV.")
    parser.add_argument("--output", required=True, help="Path to save predictions CSV.")
    args = parser.parse_args()

    logger.info(f"Loading input data from {args.data}.")
    raw_data = pd.read_csv(args.data)

    pipeline = load_pipeline(args.preprocessing, args.model, args.features)

    predictions = pipeline.run(raw_data)

    if "customerID" in raw_data.columns:
        predictions.insert(0, "customerID", raw_data["customerID"])

    logger.info(f"Saving predictions to {args.output}.")
    predictions.to_csv(args.output, index=False)
    logger.success(f"Predictions saved successfully at {args.output}.")


if __name__ == "__main__":
    logger.add(
        "./logs/core.log",
        rotation="500 MB",
        level="INFO",
        backtrace=True,
        diagnose=True,
    )
    main()
