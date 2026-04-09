# Plotting Script - Show examples of dirac features and compare dataset predictions

import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path

import random

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GT_DIR = PROJECT_ROOT / "experiments" / "CRNN_V15_ROTATION_MAE" / "groundtruth"
PRED_DIR = PROJECT_ROOT / "experiments" / "CRNN_V15_ROTATION_MAE" / "predictions"

# Comparison experiment (set COMPARE = True to enable)
COMPARE = True
PRED_DIR_B = PROJECT_ROOT / "experiments" / "CRNN_V15_ROTATION_MAE" / "predictions"
EXP_A_NAME = "CRNN_V15_MSE"
EXP_B_NAME = "CRNN_V15_MAE"

# Set to a specific scene ID to plot only that one, or None to use SCENE_FILTER / all
# e.g. SINGLE_SCENE = "scene05440"
SINGLE_SCENE = "scene05419"

# Set to a list of scene IDs to plot only those, or None for all
# e.g. SCENE_FILTER = ["scene1", "scene5", "scene12"]
SCENE_FILTER = None

# Mode: "compare" for MSE vs MAE, "get_difference" for GT vs Pred vs Error
MODE = "get_difference"  # Set to "compare" or "get_difference"

# For get_difference mode: which channels to display (indices)
# Default: [1, 2, 3, 4] = Diffuseness, Y, Z, X
# Set to None for random selection of 3 channels
DIFF_CHANNELS = None  # Set to None for random 3, or specify list like [1, 2, 3, 4]

# Channel naming
CHANNEL_NAMES = ["Mean Magnitude", "Diffuseness", "Y", "Z", "X"]


def get_channel_name(c):
    """Get human-readable channel name"""
    if c < len(CHANNEL_NAMES):
        return CHANNEL_NAMES[c]
    return f"Ch {c}"


def compute_metrics_per_channel(gt, pred):
    T, F, C = gt.shape
    mse_per_channel = []
    corr_per_channel = []

    for c in range(C):
        gt_c = gt[:, :, c].flatten()
        pred_c = pred[:, :, c].flatten()

        mse = np.mean((pred_c - gt_c) ** 2)
        mse_per_channel.append(mse)

        if np.std(gt_c) == 0 or np.std(pred_c) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(gt_c, pred_c)[0, 1]

        corr_per_channel.append(corr)

    return mse_per_channel, corr_per_channel


def print_metrics(label, gt, pred):
    mse_per_channel, corr_per_channel = compute_metrics_per_channel(gt, pred)
    C = gt.shape[2]

    print(f"\n  [{label}] Per-channel metrics:")
    print("  " + "-" * 60)
    print(f"  {'Channel':<20}{'MSE':>15}{'Correlation':>20}")
    print("  " + "-" * 60)
    for c in range(C):
        print(
            f"  {get_channel_name(c):<20}"
            f"{mse_per_channel[c]:>15.3f}"
            f"{corr_per_channel[c]:>20.3f}"
        )
    print("  " + "-" * 60)


def visualize_difference(gt_path, pred_path, channels_to_show=None):
    """
    Visualize GT, Prediction, and Absolute Error for selected channels.
    Creates one figure with 3 rows (one per channel) and 3 columns (GT, Pred, Error).
    If channels_to_show is None, randomly selects 3 channels.
    """
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    print(f"\nScene: {gt_path.stem}")
    print(f"GT shape:   {gt.shape}")
    print(f"Pred shape: {pred.shape}")

    assert gt.shape == pred.shape, "GT and prediction shape mismatch"

    if gt.ndim != 3:
        print("Expected (T, F, C)")
        return

    T, F, C = gt.shape

    if channels_to_show is None:
        # Randomly select 3 channels, excluding Mean Magnitude (channel 0)
        available_channels = [c for c in range(C) if c != 0]
        channels_to_show = sorted(random.sample(available_channels, k=min(3, len(available_channels))))
        print(f"Randomly selected channels: {channels_to_show} ({', '.join(get_channel_name(c) for c in channels_to_show)})")
    
    # Print metrics
    mse_per_channel, corr_per_channel = compute_metrics_per_channel(gt, pred)
    print("\nPer-channel metrics:")
    print("-" * 60)
    print(f"{'Channel':<20}{'MSE':>15}{'Correlation':>20}")
    print("-" * 60)
    for c in channels_to_show:
        print(
            f"{get_channel_name(c):<20}"
            f"{mse_per_channel[c]:>15.3f}"
            f"{corr_per_channel[c]:>20.3f}"
        )
    print("-" * 60)

    # Create single figure with 3 rows (channels) × 3 columns (GT, Pred, Error)
    num_channels = len(channels_to_show)
    fig, axes = plt.subplots(num_channels, 3, figsize=(16, 4.5 * num_channels),
                              gridspec_kw={'hspace': 0.4, 'wspace': 0.1})

    # Handle case of single channel (axes won't be 2D)
    if num_channels == 1:
        axes = axes.reshape(1, -1)

    for row, c in enumerate(channels_to_show):
        vmin = min(gt[:, :, c].min(), pred[:, :, c].min())
        vmax = max(gt[:, :, c].max(), pred[:, :, c].max())
        
        ch_name = get_channel_name(c)

        # Ground truth
        im = axes[row, 0].imshow(gt[:, :, c].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"GT – {ch_name}", fontsize=11)
        axes[row, 0].set_ylabel("Frequency", fontsize=9)
        axes[row, 0].set_xlabel("Time", fontsize=9)
        axes[row, 0].tick_params(labelsize=8)
        cbar1 = plt.colorbar(im, ax=axes[row, 0])
        cbar1.ax.tick_params(labelsize=7)

        # Prediction
        im = axes[row, 1].imshow(pred[:, :, c].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Prediction – {ch_name}", fontsize=11)
        axes[row, 1].set_ylabel("Frequency", fontsize=9)
        axes[row, 1].set_xlabel("Time", fontsize=9)
        axes[row, 1].tick_params(labelsize=8)
        cbar2 = plt.colorbar(im, ax=axes[row, 1])
        cbar2.ax.tick_params(labelsize=7)

        # Absolute error
        err = np.abs(pred[:, :, c] - gt[:, :, c])
        im = axes[row, 2].imshow(err.T, aspect="auto", origin="lower")
        axes[row, 2].set_title(f"Error – {ch_name}", fontsize=11)
        axes[row, 2].set_ylabel("Frequency", fontsize=9)
        axes[row, 2].set_xlabel("Time", fontsize=9)
        axes[row, 2].tick_params(labelsize=8)
        cbar3 = plt.colorbar(im, ax=axes[row, 2])
        cbar3.ax.tick_params(labelsize=7)

    plt.suptitle(f"{gt_path.stem}", fontsize=13)
    plt.tight_layout()
    plt.show()


def visualize_gt_and_pred(gt_path, pred_path, pred_path_b=None):
    gt = np.load(gt_path)
    pred = np.load(pred_path)

    print(f"\nScene: {gt_path.stem}")
    print(f"GT shape:   {gt.shape}")
    print(f"Pred shape: {pred.shape}")

    assert gt.shape == pred.shape, "GT and prediction shape mismatch"

    if gt.ndim != 3:
        print("Expected (T, F, C)")
        return

    T, F, C = gt.shape

    if pred_path_b is not None:
        pred_b = np.load(pred_path_b)
        print(f"Pred B shape: {pred_b.shape}")
        assert gt.shape == pred_b.shape, "GT and prediction B shape mismatch"
        print_metrics(EXP_A_NAME, gt, pred)
        print_metrics(EXP_B_NAME, gt, pred_b)

        for c in range(C):
            vmin = min(gt[:, :, c].min(), pred[:, :, c].min(), pred_b[:, :, c].min())
            vmax = max(gt[:, :, c].max(), pred[:, :, c].max(), pred_b[:, :, c].max())

            fig, axes = plt.subplots(2, 2, figsize=(10, 14), 
                                      gridspec_kw={'hspace': 0.8, 'wspace': 0.3})

            # Top-left: MSE model
            im = axes[0, 0].imshow(pred[:, :, c].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            axes[0, 0].set_title(f"{EXP_A_NAME} – {get_channel_name(c)}")
            axes[0, 0].set_ylabel("Frequency")
            axes[0, 0].set_xlabel("Time")
            plt.colorbar(im, ax=axes[0, 0])

            # Top-right: MAE model
            im = axes[0, 1].imshow(pred_b[:, :, c].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            axes[0, 1].set_title(f"{EXP_B_NAME} – {get_channel_name(c)}")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_xlabel("Time")
            plt.colorbar(im, ax=axes[0, 1])

            # Bottom-left: Ground truth
            im = axes[1, 0].imshow(gt[:, :, c].T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
            axes[1, 0].set_title(f"GT – {get_channel_name(c)}")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_xlabel("Time")
            plt.colorbar(im, ax=axes[1, 0])

            # Bottom-right: Error difference map
            err_diff = np.abs(pred[:, :, c] - gt[:, :, c]) - np.abs(pred_b[:, :, c] - gt[:, :, c])
            im = axes[1, 1].imshow(err_diff.T, aspect="auto", origin="lower", cmap="RdBu_r")
            axes[1, 1].set_title(f"|Err {EXP_A_NAME}| − |Err {EXP_B_NAME}|")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_xlabel("Time")
            plt.colorbar(im, ax=axes[1, 1])

            plt.suptitle(f"{gt_path.stem} – {get_channel_name(c)}")
            plt.tight_layout()
            plt.show()
    else:
        # ----------------------------
        # METRICS TABLE
        # ----------------------------
        mse_per_channel, corr_per_channel = compute_metrics_per_channel(gt, pred)

        print("\nPer-channel metrics:")
        print("-" * 60)
        print(f"{'Channel':<20}{'MSE':>15}{'Correlation':>20}")
        print("-" * 60)
        for c in range(C):
            print(
                f"{get_channel_name(c):<20}"
                f"{mse_per_channel[c]:>15.3f}"
                f"{corr_per_channel[c]:>20.3f}"
            )
        print("-" * 60)

        # ----------------------------
        # PLOTS
        # ----------------------------
        for c in range(C):
            vmin = min(gt[:, :, c].min(), pred[:, :, c].min())
            vmax = max(gt[:, :, c].max(), pred[:, :, c].max())

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(gt[:, :, c].T, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
            plt.title(f"GT – {get_channel_name(c)}")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(pred[:, :, c].T, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax)
            plt.title(f"Prediction – {get_channel_name(c)}")
            plt.xlabel("Time")
            plt.colorbar()

            plt.subplot(1, 3, 3)
            err = np.abs(pred[:, :, c] - gt[:, :, c])
            plt.imshow(err.T, aspect="auto", origin="lower")
            plt.title(f"|Error| – {get_channel_name(c)}")
            plt.xlabel("Time")
            plt.colorbar()

            plt.suptitle(f"{gt_path.stem} – {get_channel_name(c)}")
            plt.tight_layout()
            plt.show()


def get_scene_id(path):
    match = re.match(r"(scene\d+)", path.stem)
    return match.group(1) if match else None


def main():
    gt_files = sorted(GT_DIR.glob("*.npy"))
    pred_files = sorted(PRED_DIR.glob("*.npy"))

    if not gt_files:
        print(f"No groundtruth files found in {GT_DIR}")
        return

    print(f"Sample GT files: {[f.name for f in gt_files[:3]]}")
    print(f"Sample pred files: {[f.name for f in pred_files[:3]]}")

    # Build lookup tables
    pred_by_scene = {}
    for pred_path in pred_files:
        scene_id = get_scene_id(pred_path)
        if scene_id:
            pred_by_scene[scene_id] = pred_path

    pred_b_by_scene = {}
    if MODE == "compare":
        for pred_path in sorted(PRED_DIR_B.glob("*.npy")):
            scene_id = get_scene_id(pred_path)
            if scene_id:
                pred_b_by_scene[scene_id] = pred_path

    print(f"Found {len(gt_files)} scenes")
    print(f"Found {len(pred_by_scene)} predictions")
    print(f"Mode: {MODE}")
    
    skipped_no_scene_id = 0
    skipped_filtered = 0
    skipped_no_pred = 0
    skipped_no_pred_b = 0
    processed = 0

    for gt_path in gt_files:
        scene_id = get_scene_id(gt_path)

        if scene_id is None:
            skipped_no_scene_id += 1
            continue

        # Filter scenes if specified
        if SINGLE_SCENE is not None and scene_id != SINGLE_SCENE:
            skipped_filtered += 1
            continue
        if SCENE_FILTER is not None and scene_id not in SCENE_FILTER:
            skipped_filtered += 1
            continue

        pred_path = pred_by_scene.get(scene_id)
        if pred_path is None:
            skipped_no_pred += 1
            continue

        print(f"Processing {scene_id}...")
        processed += 1

        if MODE == "compare":
            pred_path_b = pred_b_by_scene.get(scene_id)
            if pred_path_b is None:
                skipped_no_pred_b += 1
                print(f"  Missing prediction B, skipping.")
                continue
            visualize_gt_and_pred(gt_path, pred_path, pred_path_b)
        elif MODE == "get_difference":
            visualize_difference(gt_path, pred_path, DIFF_CHANNELS)
        else:
            print(f"Unknown mode: {MODE}")
            return
    
    print(f"\n=== Summary ===")
    print(f"Processed: {processed}")
    print(f"Skipped (no scene ID): {skipped_no_scene_id}")
    print(f"Skipped (filtered by SINGLE_SCENE/SCENE_FILTER): {skipped_filtered}")
    print(f"Skipped (no prediction): {skipped_no_pred}")
    print(f"Skipped (no prediction B in compare mode): {skipped_no_pred_b}")


if __name__ == "__main__":
    main()