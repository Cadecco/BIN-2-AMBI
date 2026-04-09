# Path management for the project

from pathlib import Path

# ---------------------------
# Project root
# ---------------------------
ROOT = Path(__file__).resolve().parents[2]  # src/utils/ → project root

# ---------------------------
# Base directories
# ---------------------------
DATASETS_DIR = ROOT / "datasets"
CONFIGS_DIR = ROOT / "configs"
MODULES_DIR = ROOT / "modules"

# ---------------------------
# Functions for dataset-relative paths
# ---------------------------
def dataset_root(dataset_name: str) -> Path:
    """Return the root folder for a specific dataset"""
    return DATASETS_DIR / dataset_name

def features(dataset_name: str) -> Path:
    return dataset_root(dataset_name) / "features"

def binaural(dataset_name: str) -> Path:
    return dataset_root(dataset_name) / "binaural"

def FOA(dataset_name: str) -> Path:
    return dataset_root(dataset_name) / "FOA"

# ---------------------------
# Config helper
# ---------------------------
def config_path(filename: str) -> Path:
    """Return full path to a config file"""
    return CONFIGS_DIR / filename
