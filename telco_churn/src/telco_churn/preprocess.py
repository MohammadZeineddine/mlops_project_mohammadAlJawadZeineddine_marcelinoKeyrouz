import argparse
from loguru import logger
import yaml
import pandas as pd
import os
import sys
from telco_churn.data_transform import TransformFactory


def load_config(config_path):
    """
    Loads the YAML configuration for preprocessing.
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


def run_pipeline(config, data):
    """
    Runs the preprocessing pipeline on the input data.
    """
    logger.info("Starting preprocessing pipeline.")
    try:
        for step in config["data_transform"]["pipeline"]:
            transformer_name = step["transformer"]
            transformer = TransformFactory(transformer_name, **step["params"])
            columns = step["columns"]
            logger.debug(
                f"Applying transformer '{transformer_name}' on columns: {columns}"
            )
            transformed_data = transformer.fit_transform(data[columns])
            data = data.drop(columns=columns).reset_index(drop=True)
            transformed_data = transformed_data.reset_index(drop=True)
            data = pd.concat([data, transformed_data], axis=1)
        logger.success("Pipeline executed successfully.")
        return data
    except Exception as e:
        logger.error(f"Error occurred in pipeline execution: {e}")
        sys.exit(1)


def main():
    """
    Entry point for preprocessing script.
    """
    logger.info("Starting the preprocessing script.")

    # Parse command-line arguments using argparse
    parser = argparse.ArgumentParser(description="Preprocess Telco churn data")
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--data",
        default="data/raw/telco_churn.csv",
        help="Path to the input data file (default: data/raw/telco_churn.csv)",
    )
    args = parser.parse_args()

    config_path = args.config
    data_path = args.data

    # Load data
    try:
        logger.info(f"Loading data from {data_path}")
        data = pd.read_csv(data_path)
        logger.success("Data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Preprocess numeric columns
    if "TotalCharges" in data.columns:
        logger.info("Converting 'TotalCharges' to numeric.")
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    # Load and run pipeline
    config = load_config(config_path)
    processed_data = run_pipeline(config, data)

    # Save processed data
    try:
        output_path = config["global_settings"]["save_path"]
        os.makedirs(output_path, exist_ok=True)
        processed_data_path = os.path.join(output_path, "processed_data.csv")
        processed_data.to_csv(processed_data_path, index=False)
        logger.success(f"Processed data saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logger.add(
        "./logs/preprocessing.log",
        rotation="500 MB",
        level="INFO",
        backtrace=True,
        diagnose=True,
    )
    main()
