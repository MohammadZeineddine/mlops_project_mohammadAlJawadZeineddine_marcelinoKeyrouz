import argparse
import os

import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from telco_churn.data_transform import PreprocessingPipelineFactory


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
        default="config/config_preprocessing.yaml",
        help="Path to configuration file.",
    )
    args = parser.parse_args()
    main(args.config)


def main(config_path: str):
    """
    Main function for preprocessing Telco churn data and training models.
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load configuration
    logger.info("Loading configuration.")
    config = OmegaConf.load(config_path)

    # Load data
    logger.info("Loading data.")
    data = pd.read_csv(config["data_loader"]["file_path"])

    # Create preprocessing pipeline
    logger.info("Initializing preprocessing pipeline.")
    pipeline = PreprocessingPipelineFactory.create_pipeline(config["data_transform"])

    # Run preprocessing
    logger.info("Running preprocessing pipeline.")
    processed_data = pipeline.run(data)

    # Ensure the preprocessed directory exists
    processed_dir = "data/preprocessed"
    os.makedirs(processed_dir, exist_ok=True)

    # Save preprocessed data
    processed_file = os.path.join(processed_dir, "preprocessed_telco.csv")
    processed_data.to_csv(processed_file, index=False)
    logger.info(f"Preprocessed data saved to {processed_file}.")


if __name__ == "__main__":
    entrypoint()
