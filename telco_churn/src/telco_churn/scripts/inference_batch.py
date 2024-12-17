import os
import argparse
import pandas as pd
import joblib
from loguru import logger
from telco_churn.config import get_config


def load_pipeline(pipeline_path):
    """
    Load the saved preprocessing pipeline.
    """
    logger.info(f"Loading preprocessing pipeline from {pipeline_path}.")
    pipeline = joblib.load(pipeline_path)
    logger.success("Preprocessing pipeline loaded successfully.")
    return pipeline


def load_model(model_path):
    """
    Load the trained model from a file.
    """
    logger.info(f"Loading trained model from {model_path}.")
    model = joblib.load(model_path)
    logger.success("Trained model loaded successfully.")
    return model


def preprocess_data(data, pipeline):
    """
    Apply the saved preprocessing pipeline on the new data.
    Ensure the column order matches the training data.
    """
    logger.info("Applying the preprocessing pipeline on the input data.")
    transformed_data = []

    for step_name, transformer, columns in pipeline:
        if not all(col in data.columns for col in columns):
            missing_cols = [col for col in columns if col not in data.columns]
            raise ValueError(f"Missing columns in input data: {missing_cols}")

        # Apply transformation
        transformed_part = transformer.transform(data[columns])
        if hasattr(transformer, "get_feature_names_out"):
            feature_names = transformer.get_feature_names_out(columns)
        else:
            feature_names = columns

        # Append transformed data with proper column names
        transformed_part_df = pd.DataFrame(
            transformed_part, columns=feature_names, index=data.index)
        transformed_data.append(transformed_part_df)

    # Concatenate all transformed parts
    preprocessed_data = pd.concat(transformed_data, axis=1)
    logger.success("Data preprocessing completed successfully.")
    return preprocessed_data


def predict_and_save(model, data, output_path, original_data):
    """
    Use the trained model to predict outcomes and save them to a file.
    Includes the 'customerID' column for clarity in the predictions file.
    """
    logger.info("Making predictions using the trained model.")
    predictions = model.predict(data)

    # Add customerID back to the predictions
    if "customerID" in original_data.columns:
        result_df = pd.DataFrame({
            "customerID": original_data["customerID"],
            "Prediction": predictions
        })
    else:
        result_df = pd.DataFrame(predictions, columns=["Prediction"])

    # Save predictions to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    logger.success(f"Predictions saved to {output_path} with customerID.")


def main():
    """
    Main entry point for batch inference.
    """
    logger.info("Starting batch inference.")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Batch inference using a trained model and preprocessing pipeline.")
    parser.add_argument("--config", required=True,
                        help="Path to the configuration YAML file.")
    parser.add_argument("--data", required=True,
                        help="Path to the input data for prediction.")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    config, _ = get_config(config_path)

    # Load input data
    data_path = args.data
    logger.info(f"Loading input data from {data_path}.")
    data = pd.read_csv(data_path)

    # Load the preprocessing pipeline
    pipeline_path = os.path.join(
        config.output.model_dir, "preprocessing_pipeline.pkl")
    pipeline = load_pipeline(pipeline_path)

    # Preprocess the input data
    logger.info("Preprocessing input data.")
    preprocessed_data = preprocess_data(data, pipeline)

    # Load the trained model
    model_path = config.output.trained_model_path
    model = load_model(model_path)

    # Align feature names
    logger.info("Ensuring feature names align with model training.")
    expected_features = model.model.feature_names_in_
    preprocessed_data = preprocessed_data.reindex(
        columns=expected_features, fill_value=0)

    # Predict and save the results
    output_path = config.output.predictions_path
    predict_and_save(model, preprocessed_data, output_path, data)


if __name__ == "__main__":
    logger.add("./logs/inference_batch.log", rotation="500 MB",
               level="INFO", backtrace=True, diagnose=True)
    main()
