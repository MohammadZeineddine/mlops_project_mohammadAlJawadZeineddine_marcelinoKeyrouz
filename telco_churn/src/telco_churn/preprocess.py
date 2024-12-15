import yaml
import pandas as pd
import os
import sys
from src.telco_churn.data_transform.factory import transformer_factory


def load_config(config_path):
    """
    Loads the YAML configuration for preprocessing.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run_pipeline(config, data):
    """
    Runs the preprocessing pipeline on the input data.
    """
    for step in config['data_transform']['pipeline']:
        transformer = transformer_factory(
            step['transformer'], **step['params'])
        columns = step['columns']
        transformed_data = transformer.fit_transform(data[columns])
        data = data.drop(columns=columns).reset_index(drop=True)
        transformed_data = transformed_data.reset_index(drop=True)
        data = pd.concat([data, transformed_data], axis=1)
    return data


def main():
    """
    Entry point for preprocessing script.
    """
    # Load data and configuration
    if len(sys.argv) < 3:
        print("Usage: preprocess.py <config_path> <data_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    data_path = sys.argv[2]

    # Load data
    data = pd.read_csv(data_path)

    # Preprocess numeric columns
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'], errors='coerce')

    # Load and run pipeline
    config = load_config(config_path)
    processed_data = run_pipeline(config, data)

    # Save processed data
    output_path = config['global_settings']['save_path']
    os.makedirs(output_path, exist_ok=True)
    processed_data.to_csv(os.path.join(
        output_path, "processed_data.csv"), index=False)
    print(
        f"Processed data saved to {os.path.join(output_path, 'processed_data.csv')}")


if __name__ == "__main__":
    main()
