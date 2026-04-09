# Set up and run the ambisonics evaluation

from __future__ import annotations
import sys
from pathlib import Path
from evaluate_ambisonics import evaluate_dataset

# -------------------------
# CONFIG (edit these)
# -------------------------

ROOT = Path(__file__).resolve().parents[2]  # run script in project root
DATASET_NAME = "6000scenes_no_bg"          # e.g. "testset_v1"
EXPERIMENT_NAME = "CRNN_V15_STATIC_MSE"    # e.g. "unet_foa_run_03"
# Expected layout:
#   ROOT/datasets/<DATASET_NAME>/FOA/              (ground truth)
#   ROOT/experiments/<EXPERIMENT_NAME>/outputs/   (predictions)

GT_DIR = ROOT / "Datasets" / DATASET_NAME / "FOA"
PRED_RESYNTH_DIR = ROOT / "Experiments" / EXPERIMENT_NAME / "pred_resynth"
GT_RESYNTH_DIR = ROOT / "Experiments" / EXPERIMENT_NAME / "gt_resynth"
# Output CSV goes inside the experiment folder by default
OUT_CSV = ROOT / "Experiments" / EXPERIMENT_NAME / "ambiqual_per_scene.csv"
OUT_LOG = ROOT / "Experiments" / EXPERIMENT_NAME / "ambiqual_results.txt"
# Scene ID pattern: captures "scene12" from "scene12_anything.wav"
SCENE_REGEX = r"(scene\d+)_"
# AMBIQUAL params
INTENSITY_THRESHOLD = -180
ELC = 0
IGNORE_FREQ_BANDS = 0
# Where the AMBIQUAL repo is in the project
AMBIQUAL_REPO = ROOT / "modules" / "ambiqual"

# -------------------------
# Import ambiqual 
# -------------------------
sys.path.insert(0, str(AMBIQUAL_REPO))

from ambiqual import calculate_ambiqual

class LoggingPrinter:
    """Print to console and simultaneously write to a log file"""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, "w", encoding="utf-8")
    
    def print(self, *args, **kwargs):
        """Print to console and log file"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_file)
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def print_metric_table(name, stats, logger=None):
    logger.print(f"\n{name}")
    logger.print("-" * len(name))
    for key in ["mean", "std", "min", "max", "median", "p10", "p90"]:
        val = stats.get(key)
        if val is None:
            logger.print(f"{key:>8} :  None")
        else:
            logger.print(f"{key:>8} : {val: .4f}")

def main() -> None:
    logger = LoggingPrinter(OUT_LOG)
    
    try:
        logger.print("=== Running AMBIQUAL evaluation ===")
        logger.print(f"Dataset:    {DATASET_NAME}")
        logger.print(f"Experiment: {EXPERIMENT_NAME}")
        logger.print(f"GT dir:     {GT_DIR}")
        logger.print(f"Pred dir:   {PRED_RESYNTH_DIR}")
        logger.print(f"Output CSV: {OUT_CSV}")
        
        summary = evaluate_dataset(
            pred_resynth_dir=PRED_RESYNTH_DIR,
            gt_dir=GT_DIR,
            gt_resynth_dir=GT_RESYNTH_DIR,
            out_csv=OUT_CSV,
            scene_regex=SCENE_REGEX,
            calculate_ambiqual_fn=calculate_ambiqual,
            intensity_threshold=INTENSITY_THRESHOLD,
            elc=ELC,
            ignore_freq_bands=IGNORE_FREQ_BANDS,
        )
        
        logger.print("\n=== AMBIQUAL Test Set Summary ===")
        logger.print(f"Scored scenes: {summary['n_scored']} (GT indexed: {summary['n_gt_indexed']}, GT resynth indexed: {summary['n_gt_resynth_indexed']}, Pred resynth indexed: {summary['n_pred_resynth_indexed']})")
        if summary["missing_pred_resynth"]:
            mp = summary["missing_pred_resynth"]
            logger.print(f"Missing pred resynth ({len(mp)}): {mp[:10]}{'...' if len(mp) > 10 else ''}")
        if summary["missing_gt_resynth"]:
            mg = summary["missing_gt_resynth"]
            logger.print(f"Missing gt resynth ({len(mg)}): {mg[:10]}{'...' if len(mg) > 10 else ''}")
        
        print_metric_table("GT Resynth - Listening Quality (LQ)", summary["GT_Resynth_LQ"], logger)
        print_metric_table("GT Resynth - Localization Accuracy (LA)", summary["GT_Resynth_LA"], logger)
        print_metric_table("Pred Resynth - Listening Quality (LQ)", summary["Pred_Resynth_LQ"], logger)
        print_metric_table("Pred Resynth - Localization Accuracy (LA)", summary["Pred_Resynth_LA"], logger)
        print_metric_table("LQ (Pred % of GT Resynth)", summary["LQ_percent"], logger)
        print_metric_table("LA (Pred % of GT Resynth)", summary["LA_percent"], logger)
        
        logger.print(f"\nWrote per-scene CSV: {OUT_CSV}")
        logger.print(f"Wrote results log: {OUT_LOG}")
    finally:
        logger.close()

if __name__ == "__main__":
    main()