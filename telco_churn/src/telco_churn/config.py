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


def get_config(config_path):
    """
    Load configuration and determine the environment from the config file.
    """
    config = load_config(config_path)
    return config
