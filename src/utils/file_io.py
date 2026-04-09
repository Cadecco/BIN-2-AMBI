# This utility contains functions for loading and saving
# the features and associated metadata.

import os
import pickle
import torch
import pandas as pd

def save_pickle(obj, path: str):
    """
    Save an object as a pickle file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    """
    Load a pickle file.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def save_tensor(tensor, path: str):
    """
    Save a PyTorch tensor as a .pt file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor, path)

def load_tensor(path: str):
    """
    Load a PyTorch tensor from a .pt file.
    """
    return torch.load(path)

def save_metadata_csv(df, path: str):
    """
    Save a pandas DataFrame as CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_metadata_csv(path: str):
    """
    Load a pandas DataFrame from CSV.
    """
    return pd.read_csv(path)
