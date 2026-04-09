# This utiility loads .YAML configuratin files.

import yaml

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration as a Python dictionary.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config