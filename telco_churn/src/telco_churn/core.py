import argparse
import os
import sys
import pandas as pd
from loguru import logger
import joblib
from sklearn.model_selection import train_test_split
from telco_churn.config import get_config
from telco_churn.data_models import ModelFactory
from telco_churn.data_transform import TransformerFactory


def run_pipeline(config, data):
    """
    Runs the preprocessing pipeline on the input data.
    """
    logger.info("Starting preprocessing pipeline.")
    try:
        logger.debug(
            f"Columns before transformation: {data.columns.tolist()}"
        )  # Log initial columns

        for step in config.data_transform.pipeline:
            transformer_name = step.transformer
            transformer = TransformerFactory.get_transformer(
                transformer_name, **step.params
            )
            columns = step.columns

            # Check if the required columns exist in the data
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing columns in data: {missing_columns}")
                sys.exit(1)

            logger.debug(
                f"Applying transformer '{transformer_name}' on columns: {columns}"
            )
            transformed_data = transformer.fit_transform(data[columns])

            # Drop the transformed columns and reset index
            data = data.drop(columns=columns).reset_index(drop=True)
            transformed_data = transformed_data.reset_index(drop=True)

            # Concatenate the transformed data back to the original data
            data = pd.concat([data, transformed_data], axis=1)

        logger.success("Pipeline executed successfully.")
        return data

    except Exception as e:
        logger.error(f"Error occurred in pipeline execution: {e}")
        sys.exit(1)


def train_model(config, X_train, y_train, X_test, y_test):
    """
    Train a machine learning model using the given configuration and data.
    """
    model_name = config.model.name
    model_params = config.model.get("params", {})
    logger.info(f"Creating model '{model_name}' with parameters: {model_params}")
    try:
        model = ModelFactory(model_name, **model_params)
        logger.success(f"Model '{model_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)

    # Train the model
    logger.info("Training the model...")
    try:
        model.train(X_train, y_train)
        logger.success("Model trained successfully.")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        sys.exit(1)

    # Evaluate the model
    logger.info("Evaluating the model...")
    try:
        metrics = model.evaluate(X_test, y_test)
        logger.success(f"Model evaluated successfully. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        sys.exit(1)

    # Save the model
    output_dir = config.output.model_dir
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    try:
        joblib.dump(model, model_path)
        logger.success(f"Model saved at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for training and preprocessing based on config.
    """
    logger.info("Starting the pipeline.")

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Preprocess and Train a machine learning model."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # Load configuration using get_config from config.py
    config_path = args.config
    config, environment = get_config(config_path)

    # Load preprocessed dataset
    data_path = config.data.path
    try:
        logger.info(f"Loading dataset from {data_path}")
        data = pd.read_csv(data_path)
        logger.success("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Preprocess the data if necessary
    if "data_transform" in config:
        data = run_pipeline(config, data)

    # Separate features and target
    target_column = config.data.target_column
    logger.info(f"Separating features and target column '{target_column}'")
    X = data.drop(columns=[target_column, "customerID"])
    y = data[target_column]

    # Split dataset
    test_size = config.training.test_size
    random_state = config.training.random_state
    logger.info("Splitting dataset into training and testing sets.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.success("Dataset split successfully.")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        sys.exit(1)

    # Train the model
    train_model(config, X_train, y_train, X_test, y_test)

    # Save the processed data (optional) with environment-specific name
    if "global_settings" in config and "save_path" in config.global_settings:
        processed_data_path = os.path.join(
            config.global_settings.save_path, f"processed_data_{environment}.csv"
        )
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        try:
            data.to_csv(processed_data_path, index=False)
            logger.success(f"Processed data saved to {processed_data_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            sys.exit(1)


if __name__ == "__main__":
    logger.add(
        "./logs/pipeline.log",
        rotation="500 MB",
        level="INFO",
        backtrace=True,
        diagnose=True,
    )
    main()
