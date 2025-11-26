import json
import sys
from pathlib import Path
import hashlib

import numpy as np
import torch


class Config:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # If the value is a nested dictionary, create a Config object for it
                setattr(self, key, Config(value))
            else:
                # Set the attribute with the key and value
                setattr(self, key, value)

    def __repr__(self, level=0):
        # Create a string representation of the Config object
        config_str = ""
        for key, value in vars(self).items():
            if isinstance(value, Config):
                # Recursively call __repr__ for nested Config objects
                config_str += f"{'  ' * level}{key}:\n{value.__repr__(level + 1)}"
            else:
                config_str += f"{'  ' * level}{key}={value}\n"
        return config_str


def set_seed(seed):
    """set random seed."""
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_files(parquet_dirs, valid_ratio):
    total_num_excluded_last = len(parquet_dirs) - 1
    threshold = int(np.floor(total_num_excluded_last * (1 - valid_ratio)))
    train_files = parquet_dirs[:threshold]
    test_files = parquet_dirs[threshold:]
    return train_files, test_files


def save_config(config, save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4, default=str)  # Use default=str for Path objects


def make_hash(config):
    """
    Generate a content-based ID maker for a config dict of hyperparams.
    Args:
        hyperparams (dict): A dict of hyperparams.
                            Paths are converted to strings for serialization.
    Returns:
        str: A SHA-256 hash of the hyperparams.
    """
    # Serializes the dict to a deterministic JSON string
    config_str = json.dumps(config, sort_keys=True, default=str)

    # A SHA-256 hex digest of the string
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    return config_hash
