import os
import sys

from loguru import logger


def train():
    logger.info("Starting training process...")
    os.system(
        f"python src/telco_churn/scripts/train_batch.py --config {os.getenv('CONFIG_PATH')}")


def inference():
    logger.info("Starting inference process...")
    os.system(
        f"python src/telco_churn/scripts/inference_batch.py "
        f"--config {os.getenv('CONFIG_PATH')} --data {os.getenv('DATA_PATH')}"
    )


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "help"
    if task == "train":
        train()
    elif task == "inference":
        inference()
    else:
        print("Usage: docker run <image_name> [train|inference]")
