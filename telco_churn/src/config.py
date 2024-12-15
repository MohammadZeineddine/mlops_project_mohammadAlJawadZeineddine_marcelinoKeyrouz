import os
from typing import Union
from omegaconf import OmegaConf
from pydantic import BaseModel, validator
from typing import List, Dict


class DataLoaderConfig(BaseModel):
    file_path: str
    file_type: str

from typing import Union, List, Dict
from pydantic import BaseModel, validator

from typing import Union, List, Dict
from pydantic import BaseModel, field_validator

class DataTransformPipeline(BaseModel):
    name: str
    params: Dict[str, Union[List[str], str]]  # Allow either a list of strings or a single string for params

    @field_validator('params')
    def validate_params(cls, value):
        if isinstance(value, dict):
            # Check for 'strategy' key and ensure it is either a string or a list of strings
            if 'strategy' in value:
                strategy = value['strategy']
                if isinstance(strategy, str):
                    # If it's a string, convert it into a list of strings
                    value['strategy'] = [strategy]
                elif isinstance(strategy, list) and all(isinstance(i, str) for i in strategy):
                    pass  # Valid list of strings
                else:
                    raise ValueError(f"Invalid 'strategy' value: {strategy}. Must be a string or list of strings.")
        return value





class DataTransformConfig(BaseModel):
    pipeline: List[DataTransformPipeline]

    @validator('pipeline')
    def check_pipeline_names(cls, value):
        valid_names = {'imputer', 'encoder', 'scaler'}
        for item in value:
            if item.name not in valid_names:
                raise ValueError(f"Invalid pipeline name: {item.name}. Must be one of {valid_names}.")
        return value


def load_config(env: str = None, config_path: str = None):
    """
    Load and validate the configuration based on the environment or a full path.
    Args:
        env (str): The environment name (dev, test, production).
        config_path (str): The full path to the configuration file.
    """
    # If a full path is provided, load directly from that path
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        config = OmegaConf.load(config_path)
    # Otherwise, load based on the environment name
    elif env:
        config_path = f"config/config_{env}.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file for environment '{env}' not found.")
        config = OmegaConf.load(config_path)
    else:
        raise ValueError("Either 'env' or 'config_path' must be provided.")

    # Validate the configuration
    data_loader = DataLoaderConfig(**config.data_loader)
    data_transform = DataTransformConfig(**config.data_transform)

    return data_loader, data_transform
