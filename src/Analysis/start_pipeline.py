# This program starts the dataset geneartion and features analysis pipeline.
# As dataset generation starts the binaural and DirAC analysis programs are launched 
# which begin extracting input and groundtruth features in parallel.

from pathlib import Path
import yaml
import time
from multiprocessing import Process, Queue

from generator import generate_features
from feature_extractor import FeatureExtractor
from dirac_analyser import DiracAnalyser;

if __name__ == "__main__":
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
    DATASET_DIR = ROOT / "datasets" / dataset_name
    BIN_DIR = DATASET_DIR / config["paths"]["binaural_folder"]
    AMBI_DIR = DATASET_DIR / config["paths"]["ambisonics_folder"]
    
    FEATURES_DIR = SAVE_ROOT / "datasets" / dataset_name / config["paths"]["features_folder"]
    GT_DIR = SAVE_ROOT / "datasets" / dataset_name / config["paths"]["groundtruth_folder"]
    FFT_DIR = ROOT / "datasets" / dataset_name / config["paths"]["fft_folder"]

    # Create folders if they don't exist
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    AMBI_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    FFT_DIR.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config file used to the dataset folder:
    with (DATASET_DIR / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Initialise queues for binaural and ambisonics:
    q_bin = Queue()
    q_ambi = Queue()

    # Instantiate the binaural and dirac feature extractors
    binaural_analyser = FeatureExtractor(config)
    dirac_analyser = DiracAnalyser(config)    

    # Wrap run functions in multiprocessing
    # Pass arguments separately to avoid starting the generator yet
    p_gen = Process(target=generate_features, args=(config, q_bin, q_ambi))
    p_analysis = Process(target=binaural_analyser.run, args=(q_bin, 'pipe'))
    p_dirac = Process(target=dirac_analyser.run, args=(q_ambi, 'pipe'))

    # Start processes
    #--------------------------------------------------------------------------------------
    start = time.time()
    p_gen.start()
    p_analysis.start()
    p_dirac.start()

    # Wait for all to finish
    #p_gen.join()
    p_analysis.join()
    p_dirac.join()

    print(f"✔ Process Finished!")
    end = time.time()
    print(f"Pipeline finished in {end - start:.2f} seconds")