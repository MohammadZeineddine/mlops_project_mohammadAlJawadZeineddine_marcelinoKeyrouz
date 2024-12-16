import os
import sys
from omegaconf import OmegaConf
from loguru import logger


def load_config(config_path):
    """
    Load the configuration from a YAML file using OmegaConf.
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        config = OmegaConf.load(config_path)
        logger.success("Configuration loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def get_environment_from_config(config_path):
    """
    Extracts the environment type (dev, test, etc.) from the config file name.
    """
    base_name = os.path.basename(config_path)
    if "dev" in base_name:
        return "dev"
    elif "test" in base_name:
        return "test"
    else:
        return "default"


def get_config(config_path):
    """
    Load configuration and determine the environment from the config file.
    """
    config = load_config(config_path)
    environment = get_environment_from_config(config_path)
    return config, environment
