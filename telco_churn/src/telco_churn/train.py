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
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    """
    Train a machine learning model using a configuration file.
    """
    # Load training configuration
    if len(sys.argv) < 2:
        print("Usage: train.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    # Load preprocessed dataset
    data_path = config["data"]["path"]
    data = pd.read_csv(data_path)

    # Separate features and target
    target_column = config["data"]["target_column"]
    X = data.drop(columns=[target_column, "customerID"])
    y = data[target_column]

    # Split dataset
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    # Create model
    model_name = config["model"]["name"]
    model_params = config["model"].get("params", {})
    model = model_factory(model_name, **model_params)

    # Train model
    print("Training the model...")
    model.train(X_train, y_train)

    # Evaluate model
    print("Evaluating the model...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Evaluation Metrics: {metrics}")

    # Save the model
    output_dir = config["output"]["model_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    main()
