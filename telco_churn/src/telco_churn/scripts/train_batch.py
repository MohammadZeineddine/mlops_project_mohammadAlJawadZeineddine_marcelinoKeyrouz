import os
import argparse
import pandas as pd
import numpy as np
import mlflow
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from telco_churn.config import get_config
from telco_churn.data_models import ModelFactory
from telco_churn.data_transform import TransformerFactory


def run_pipeline(config, data):
    """
    Executes the preprocessing pipeline on the input data.
    """
    logger.info("Starting preprocessing pipeline.")
    for step in config.data_transform.pipeline:
        transformer_name = step.transformer
        transformer = TransformerFactory.get_transformer(
            transformer_name, **step.params)
        columns = step.columns

        # Check for missing columns
        missing_columns = [col for col in columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        # Apply transformations
        transformed_data = transformer.fit_transform(data[columns])
        data = data.drop(columns=columns).reset_index(drop=True)
        transformed_data = transformed_data.reset_index(drop=True)
        data = pd.concat([data, transformed_data], axis=1)

    logger.success("Pipeline executed successfully.")
    return data


def train_and_log_model(config, X_train, y_train, X_test, y_test):
    """
    Trains a model and logs parameters, metrics, and artifacts to MLflow.
    """
    model_name = config.model.name
    model_params = config.model.get("params", {})
    logger.info(
        f"Creating model '{model_name}' with parameters: {model_params}")
    model = ModelFactory(model_name, **model_params)

    # MLflow experiment setup
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(model_params)

        # Train the model
        logger.info("Training the model...")
        model.train(X_train, y_train)

        # Evaluate the model
        logger.info("Evaluating the model...")
        metrics = model.evaluate(X_test, y_test)

        # Filter metrics for MLflow logging (scalars only)
        scalar_metrics = {k: float(v) for k, v in metrics.items(
        ) if isinstance(v, (int, float, np.float64))}
        mlflow.log_metrics(scalar_metrics)

        # Log non-scalar metrics as artifacts or skip
        if "confusion_matrix" in metrics:
            confusion_matrix_path = os.path.join(
                config.output.model_dir, "confusion_matrix.csv")
            pd.DataFrame(metrics["confusion_matrix"]).to_csv(
                confusion_matrix_path, index=False)
            mlflow.log_artifact(confusion_matrix_path)
            logger.info("Logged confusion matrix as an artifact.")

        # Save and log the model
        output_dir = config.output.model_dir
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        logger.success(f"Model saved and logged to MLflow at {model_path}")


def main():
    """
    Main entry point for training and logging with MLflow.
    """
    logger.info("Starting batch training with MLflow.")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a model and log to MLflow.")
    parser.add_argument("--config", required=True,
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    config, environment = get_config(config_path)

    # Load dataset
    data_path = config.data.path
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_csv(data_path)

    # Check if preprocessing is needed
    if "data_transform" in config:
        logger.info("Running preprocessing pipeline...")
        data = run_pipeline(config, data)
    else:
        logger.info("Using preprocessed data.")

    # Separate features and target
    target_column = config.data.target_column
    logger.info(f"Separating features and target column '{target_column}'")
    X = data.drop(columns=[target_column, "customerID"])
    y = data[target_column]

    # Split the dataset
    logger.info("Splitting dataset into training and testing sets.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.training.test_size, random_state=config.training.random_state
    )

    # Train the model and log to MLflow
    train_and_log_model(config, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    logger.add("./logs/train_batch.log", rotation="500 MB",
               level="INFO", backtrace=True, diagnose=True)
    main()
