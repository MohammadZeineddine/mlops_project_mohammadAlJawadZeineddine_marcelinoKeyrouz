FROM python:3.11-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir pipx && \
    pipx install poetry && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

COPY poetry.lock pyproject.toml /app/

RUN poetry install & \
    poetry run pip install uvicorn

COPY . /app

RUN poetry run pip install --timeout=120 -e .

EXPOSE 8000

ENTRYPOINT ["poetry", "run"]

CMD ["uvicorn", "src.telco_churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
