# This program starts the analysis of the input
# binaural .wav files in the dataset:

from pathlib import Path
import yaml
from feature_extractor import FeatureExtractor

# Get project root directory
ROOT = Path(__file__).resolve().parents[2]

# Load chosen config file
config_file = "perceptual_config.yaml"
CONFIG_PATH = ROOT / "configs" / config_file

with CONFIG_PATH.open("r") as f:
    config = yaml.safe_load(f)

# Store features directly to Google Drive?
if config["paths"].get("save_to_drive", False):
    drive_path = Path(config["paths"].get("google_drive_path", ""))
    SAVE_ROOT = drive_path
else:
    SAVE_ROOT = ROOT
            
# Build dataset paths
dataset_name = config["dataset_name"]
FEATURES_DIR = SAVE_ROOT / "datasets" / dataset_name / config["paths"]["features_folder"]

# Create folders if they don't exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate an extractor and run it to extract 
# binaural features. Choose between 'pipe' or 'test' modes.
extractor = FeatureExtractor(config)
extractor.run(mode = 'test')

print(f"✔ Analysis Finished: {FEATURES_DIR}")