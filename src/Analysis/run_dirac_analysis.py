# This program starts the dirac analysis of the ambisonics data
# add stores it as groundtruth data.

from pathlib import Path
import yaml
from dirac_analyser import DiracAnalyser

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
GT_DIR = SAVE_ROOT / "datasets" / dataset_name / config["paths"]["groundtruth_folder"]
FFT_DIR = SAVE_ROOT / "datasets" / dataset_name / config["paths"]["fft_folder"]

# Create folders if they don't exist
GT_DIR.mkdir(parents=True, exist_ok=True)
FFT_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate an extractor and run it to extract 
# binaural features
analyser = DiracAnalyser(config)
analyser.run(mode = 'test')

print(f"✔ Analysis Finished: {GT_DIR}")