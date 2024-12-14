# src/main.py
import argparse
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf
from data_transform.factory import PreprocessingPipelineFactory


def main(config_path: str):
    # Load configuration
    logger.info("Loading configuration.")
    config = OmegaConf.load(config_path)

    # Load data
    logger.info("Loading data.")
    data = pd.read_csv(config["data_loader"]["file_path"])

    # Create preprocessing pipeline
    logger.info("Initializing preprocessing pipeline.")
    pipeline = PreprocessingPipelineFactory.create_pipeline(
        config["data_transform"])

    # Run preprocessing
    logger.info("Running preprocessing pipeline.")
    processed_data = pipeline.run(data)

    # Save preprocessed data
    processed_file = "data/preprocessed/preprocessed_telco.csv"
    processed_data.to_csv(processed_file, index=False)
    logger.info(f"Preprocessed data saved to {processed_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn Preprocessing")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file.")
    args = parser.parse_args()
    main(args.config)
