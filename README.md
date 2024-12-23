# Telco Customer Churn Prediction

This project implements a machine learning pipeline to predict customer churn using the Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code). The project includes training, inference, and deployment components, with support for Docker and CI/CD workflows.

---

## Features
- **Training**: Train models using configurable pipelines (e.g., Random Forest, Logistic Regression).
- **Inference**: Make predictions on new data using trained models.
- **API**: Expose the model via a FastAPI application with endpoints for predictions and batch processing.
- **Docker**: Build and run the application in a containerized environment.
- **CI/CD**: Automated workflows for linting, testing, and formatting using GitHub Actions.

---

## Directory Structure

```
├── config
│   ├── config_inference.yaml        # Configuration for inference
│   ├── config_train_dev.yaml        # Training config for development
│   └── config_train_prod.yaml       # Training config for production
├── data
│   └── raw
│       ├── new_data.csv             # Sample inference data
│       └── telco_churn.csv          # Dataset for training
├── logs                             # Logs for MLflow and other processes
├── models                           # Saved preprocessing pipelines and models
├── mlruns                           # MLflow artifacts and run tracking
├── src
│   ├── telco_churn
│   │   ├── api.py                   # FastAPI application
│   │   ├── config.py                # Configuration loader
│   │   ├── data_models              # Model implementations
│   │   ├── data_transform           # Preprocessing transformers
│   │   └── scripts                  # Training and inference scripts
├── tests                            # Unit tests for components
├── Dockerfile                       # Docker build file
├── docker-compose.yml               # Docker Compose configuration
├── run_app.py                       # Script to run the FastAPI application
└── README.md                        # Project documentation
```

---

## Setup and Installation

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MohammadZeineddine/mlops_project_mohammadAlJawadZeineddine_marcelinoKeyrouz.git
   cd mlops_project_mohammadAlJawadZeineddine_marcelinoKeyrouz\telco_churn\
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Run MLflow UI locally:
   ```bash
   poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

### Docker Setup
1. Build the Docker image:
   ```bash
   docker build -t telco-churn .
   ```

2. Run training and inference with Docker Compose:
   ```bash
   docker-compose --env-file .env.dev run --rm train
   docker-compose --env-file .env.prod run --rm train
   docker-compose run --rm inference
   ```

---

## Training

Train models using predefined configurations:

### Development Configuration
```bash
poetry run train-telco --config config/config_train_dev.yaml
```

### Production Configuration
```bash
poetry run train-telco --config config/config_train_prod.yaml
```

---

## Inference

Make predictions on new data:

### Run Locally
```bash
poetry run inference-pipeline
```

### Using the API
Run the FastAPI application:
```bash
poetry run python run_app.py --env dev
```

Access the API at `http://localhost:8000`. Available endpoints:
- `/process`: Accepts JSON or file uploads for predictions.
- `/healthcheck`: API status check.

---

## CI/CD

This project uses GitHub Actions for Continuous Integration and Deployment:

### Workflows
- **Linting and Formatting**:
  ```bash
  poetry run invoke lint
  poetry run invoke format
  ```
- **Testing**:
  Uncomment the testing step in `.github/workflows/ci.yml`:
  ```yaml
  # - name: Run Tests
  #   run: poetry run invoke test
  ```
- **Typing Checks**:
  Uncomment the typing step in `.github/workflows/ci.yml`:
  ```yaml
  # - name: Run Typing Checks
  #   run: poetry run invoke type
  ```

---

## Monitoring and Logging
- MLflow is used for experiment tracking and artifact management.
- Logs are saved to the `logs/` directory.

---

## Future Enhancements
- **Model Registry**: Use MLflow Model Registry to manage production-ready models.
- **Deployment**: Automate deployment pipelines for serving models.
- **Monitoring**: Add metrics tracking and alerting using Prometheus or similar tools.
- **Jenkins Integration**: Add Jenkins pipelines for build automation and CI/CD workflows.

---

## Contributors
- **Mohammad Al Jawad Zeineddine**
- **Marcelino Keyrouz**

