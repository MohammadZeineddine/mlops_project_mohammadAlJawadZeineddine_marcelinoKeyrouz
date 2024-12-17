from telco_churn.core import load_pipeline
import pandas as pd

if __name__ == "__main__":
    PREPROCESSING_PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
    TRAINED_MODEL_PATH = "models/logistic_regression_model.pkl"
    FEATURE_NAMES_PATH = "models/feature_names.txt"
    NEW_DATA_PATH = "data/raw/new_data.csv"
    OUTPUT_PATH = "output/predictions_with_customerID.csv"

    # Load new data
    raw_data = pd.read_csv(NEW_DATA_PATH)

    # Load pipeline
    pipeline = load_pipeline(PREPROCESSING_PIPELINE_PATH,
                             TRAINED_MODEL_PATH, FEATURE_NAMES_PATH)

    # Run the pipeline
    predictions = pipeline.run(raw_data)

    # Add customerID for reporting
    if "customerID" in raw_data.columns:
        predictions.insert(0, "customerID", raw_data["customerID"])

    # Save predictions
    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")
