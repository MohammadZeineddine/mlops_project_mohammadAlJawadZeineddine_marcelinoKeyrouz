import os

import pandas as pd
from loguru import logger

from telco_churn.core import load_pipeline


def main():
    """
    Main function for running the inference pipeline.
    """
    PREPROCESSING_PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
    TRAINED_MODEL_PATH = "models/logistic_regression_model.pkl"
    FEATURE_NAMES_PATH = "models/feature_names.txt"
    NEW_DATA_PATH = "data/raw/new_data.csv"
    OUTPUT_PATH = "output/predictions_with_customerID.csv"

    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(OUTPUT_PATH)
        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        # Load new data
        logger.info("Loading new data.")
        raw_data = pd.read_csv(NEW_DATA_PATH)

        # Load pipeline
        logger.info("Loading preprocessing pipeline and trained model.")
        pipeline = load_pipeline(
            PREPROCESSING_PIPELINE_PATH, TRAINED_MODEL_PATH, FEATURE_NAMES_PATH
        )

        # Run the pipeline
        logger.info("Running the inference pipeline.")
        predictions = pipeline.run(raw_data)

        # Add customerID for reporting
        if "customerID" in raw_data.columns:
            predictions.insert(0, "customerID", raw_data["customerID"])

        # Save predictions
        logger.info(f"Saving predictions to {OUTPUT_PATH}.")
        predictions.to_csv(OUTPUT_PATH, index=False)
        logger.success(f"Predictions saved to {OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    main()
