import joblib
import pandas as pd


def main():
    # Load the saved model
    model_path = "models/logistic_regression_model.pkl"
    model = joblib.load(model_path)

    # Load new data for prediction (preprocessed)
    # Ensure this is preprocessed
    new_data = pd.read_csv("data/processed/new_data.csv")

    # Make predictions
    predictions = model.predict(new_data)
    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
