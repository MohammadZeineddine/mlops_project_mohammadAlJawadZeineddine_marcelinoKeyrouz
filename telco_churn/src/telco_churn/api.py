import os
from typing import List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from telco_churn.config import get_config

app = FastAPI()

pipeline = None
model = None
expected_features = []

load_dotenv()

config_path = os.getenv("CONFIG_PATH", "config/config_train_dev.yaml")

class PredictionInput(BaseModel):
    customerID: str
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    tenure: int
    TotalCharges: Optional[float] = None

class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

@app.on_event("startup")
def load_artifacts():
    """
    Load the preprocessing pipeline and trained model at application startup.
    """
    global pipeline, model, expected_features

    logger.info("Loading configuration and artifacts.")

    try:
        config = get_config(config_path)

        pipeline_path = os.path.join(config.output.model_dir, "preprocessing_pipeline.pkl")
        pipeline = joblib.load(pipeline_path)
        logger.success("Preprocessing pipeline loaded successfully.")

        model_path = config.output.trained_model_path
        model = joblib.load(model_path)
        logger.success("Trained model loaded successfully.")

        expected_features = model.model.feature_names_in_

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise RuntimeError(f"Failed to load artifacts: {e}")

@app.post("/process")
async def process(input_data: BatchPredictionInput = None, file: UploadFile = File(None)):
    """
    Unified endpoint to handle predictions and file uploads.
    - If data is provided, performs prediction.
    - If a file is uploaded, processes the file for batch processing.
    """
    if input_data:
        try:
            logger.info("Received data for prediction.")

            data = pd.DataFrame([item.dict() for item in input_data.data])

            customer_ids = data["customerID"] if "customerID" in data.columns else None

            transformed_data = []
            for step_name, transformer, columns in pipeline:
                if not all(col in data.columns for col in columns):
                    missing_cols = [col for col in columns if col not in data.columns]
                    raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

                transformed_part = transformer.transform(data[columns])

                if hasattr(transformer, "get_feature_names_out"):
                    feature_names = transformer.get_feature_names_out(columns)
                else:
                    feature_names = columns

                transformed_part_df = pd.DataFrame(transformed_part, columns=feature_names, index=data.index)
                transformed_data.append(transformed_part_df)

            preprocessed_data = pd.concat(transformed_data, axis=1)

            preprocessed_data = preprocessed_data.reindex(columns=expected_features, fill_value=0)

            predictions = model.predict(preprocessed_data)
            response = {"predictions": predictions.tolist()}

            if customer_ids is not None:
                response["customerID"] = customer_ids.tolist()

            return response

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

    elif file:
        try:
            logger.info("File upload received.")
            df = pd.read_csv(file.file)
            if df.empty:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            logger.success("File processed successfully.")
            return {"message": "File processed successfully.", "columns": df.columns.tolist()}

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="No input data or file provided.")

@app.get("/healthcheck")
def healthcheck():
    """
    Healthcheck endpoint to verify API status.
    """
    return {"status": "API is running"}
