# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Install pipx and Poetry in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir pipx && \
    pipx install poetry && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy only dependency files first to leverage Docker's caching
COPY poetry.lock pyproject.toml /app/

# Install dependencies with Poetry
RUN poetry install --no-root --no-cache

# Install uvicorn for serving FastAPI
RUN poetry run pip install uvicorn

# Copy the rest of the project files
COPY . /app

# Explicitly install the project in editable mode
RUN poetry run pip install -e .

# Set Python's PATH for Poetry's virtual environment
ENV PATH="/root/.local/bin:$PATH"

# Expose the FastAPI default port (8000)
EXPOSE 8000

# Entrypoint for running commands dynamically
ENTRYPOINT ["poetry", "run"]

# Default command (runs FastAPI with uvicorn)
CMD ["uvicorn", "src.telco_churn.main:app", "--host", "0.0.0.0", "--port", "8000"]
