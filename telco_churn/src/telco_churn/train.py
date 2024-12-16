from loguru import logger
import os
import sys
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from models.factory import model_factory


def load_config(config_path):
    """
    Load the training configuration from a YAML file.
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.success("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def main():
    """
    Train a machine learning model using a configuration file.
    """
    logger.info("Starting the training script.")
    # Load training configuration
    if len(sys.argv) < 2:
        logger.error("Usage: train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    # Load preprocessed dataset
    data_path = config["data"]["path"]
    try:
        logger.info(f"Loading dataset from {data_path}")
        data = pd.read_csv(data_path)
        logger.success("Dataset loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load dataset, data is not processed: {e}")
        sys.exit(1)

    # Separate features and target
    target_column = config["data"]["target_column"]
    logger.info(f"Separating features and target column '{target_column}'")
    X = data.drop(columns=[target_column, "customerID"])
    y = data[target_column]

    # Split dataset
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    logger.info("Splitting dataset into training and testing sets.")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.success("Dataset split successfully.")
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        sys.exit(1)

    # Create model
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    logger.info(f"Creating model '{model_name}' with parameters: {model_params}")
    try:
        model = model_factory(model_name, **model_params)
        logger.success(f"Model '{model_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        sys.exit(1)

    # Train model
    logger.info("Training the model...")
    try:
        model.train(X_train, y_train)
        logger.success("Model trained successfully.")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        sys.exit(1)

    # Evaluate model
    logger.info("Evaluating the model...")
    try:
        metrics = model.evaluate(X_test, y_test)
        logger.success(f"Model evaluated successfully. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        sys.exit(1)

    # Save the model
    output_dir = config["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    try:
        joblib.dump(model, model_path)
        logger.success(f"Model saved at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.add("./logs/training.log", rotation="500 MB", level="INFO", backtrace=True, diagnose=True)
    main()
