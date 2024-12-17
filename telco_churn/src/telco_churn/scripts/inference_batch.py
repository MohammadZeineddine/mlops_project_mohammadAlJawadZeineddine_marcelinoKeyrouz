import os
import argparse
import pandas as pd
import joblib
from loguru import logger
from telco_churn.config import get_config
from telco_churn.data_transform import TransformerFactory
import joblib


def save_pipeline(pipeline, output_dir):
    """
    Save the fitted preprocessing pipeline.
    """
    pipeline_path = os.path.join(output_dir, "preprocessing_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    logger.success(f"Preprocessing pipeline saved to {pipeline_path}.")
    return pipeline_path


def load_pipeline(pipeline_path):
    """
    Load the saved preprocessing pipeline.
    """
    logger.info(f"Loading preprocessing pipeline from {pipeline_path}.")
    pipeline = joblib.load(pipeline_path)
    logger.success("Preprocessing pipeline loaded successfully.")
    return pipeline


def run_pipeline(config, data, pipeline):
    """
    Executes the saved preprocessing pipeline on the input data.
    """
    logger.info("Applying saved preprocessing pipeline.")
    transformed_data = pipeline.transform(data)
    logger.success("Data transformed successfully.")
    return transformed_data


def predict_and_save(model, data, output_path):
    """
    Use the trained model to predict outcomes and save them to a file.
    """
    logger.info("Making predictions.")
    predictions = model.predict(data)
    prediction_df = pd.DataFrame(predictions, columns=["Prediction"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prediction_df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path}.")


def validate_config(config):
    """
    Validates the configuration file for required keys.
    """
    required_keys = ["output.model_dir",
                     "output.trained_model_path", "output.predictions_path"]
    for key in required_keys:
        if not config.get(key, None):
            raise ValueError(f"Missing required key in configuration: {key}")


def main():
    """
    Main entry point for batch inference.
    """
    logger.info("Starting batch inference.")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Batch inference using a trained model.")
    parser.add_argument("--config", required=True,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--data", required=True,
                        help="Path to the input data for prediction.")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    config, environment = get_config(config_path)
    validate_config(config)

    # Load input data
    data_path = args.data
    logger.info(f"Loading input data from {data_path}.")
    data = pd.read_csv(data_path)

    # Load preprocessing pipeline
    pipeline_path = os.path.join(
        config.output.model_dir, "preprocessing_pipeline.pkl")
    pipeline = load_pipeline(pipeline_path)

    # Apply the pipeline
    data = run_pipeline(config, data, pipeline)

    # Drop unnecessary columns like "customerID"
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # Load the trained model
    model_path = config.output.trained_model_path
    model = load_model(model_path)

    # Make predictions and save results
    predict_and_save(model, data, config.output.predictions_path)


if __name__ == "__main__":
    logger.add("./logs/inference_batch.log", rotation="500 MB",
               level="INFO", backtrace=True, diagnose=True)
    main()
