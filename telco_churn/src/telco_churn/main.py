import argparse
import os
import pandas as pd
from loguru import logger
from telco_churn.data_transform import PreprocessingPipelineFactory
from config import load_config  # Import the load_config function from config.py


def entrypoint():
    """
    Entry point for the script, used when running with Poetry.
    Handles command-line argument parsing and calls the main function.
    """
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn Preprocessing and Model Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dev",  # Default to dev environment
        help="Environment (dev, test, production) for configuration file or full path to configuration file.",
    )
    args = parser.parse_args()
    main(args.config)


def main(config_env_or_path: str):
    """
    Main function for preprocessing Telco churn data and training models.
    Args:
        config_env_or_path (str): Environment name (dev, test, production) or full path to configuration file.
    """
    # Check if the argument is a full path or an environment name
    if os.path.exists(config_env_or_path):
        # If it is a file path, load the file directly
        config_path = config_env_or_path
        logger.info(f"Loading configuration from {config_path}.")
        data_loader, data_transform = load_config(
            config_path=config_path
        )  # Load config from file path
    else:
        # Otherwise, treat it as an environment name
        logger.info(f"Loading {config_env_or_path} environment configuration.")
        data_loader, data_transform = load_config(
            env=config_env_or_path
        )  # Load config for environment

    # Load data
    logger.info("Loading data.")
    data = pd.read_csv(data_loader.file_path)  # Use validated file_path

    # Create preprocessing pipeline
    logger.info("Initializing preprocessing pipeline.")
    pipeline = PreprocessingPipelineFactory.create_pipeline(
        data_transform.dict()
    )  # Pass validated data_transform

    # Run preprocessing
    logger.info("Running preprocessing pipeline.")
    processed_data = pipeline.run(data)

    # Ensure the preprocessed directory exists
    processed_dir = "data/preprocessed"
    os.makedirs(processed_dir, exist_ok=True)

    # Determine the processed file name based on environment
    if config_env_or_path == "dev":
        processed_file = os.path.join(processed_dir, "preprocessed_telco_dev.csv")
    elif config_env_or_path == "prod":
        processed_file = os.path.join(processed_dir, "preprocessed_telco_prod.csv")
    else:
        processed_file = os.path.join(processed_dir, "preprocessed_telco.csv")

    # Save preprocessed data
    processed_data.to_csv(processed_file, index=False)
    logger.info(f"Preprocessed data saved to {processed_file}.")


if __name__ == "__main__":
    entrypoint()
