# Plotting Script - Plot an example of input features to check them.

import os
import torch
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "datasets", "6000scenes_no_bg", "features")

def visualize_feature_file(file_path):
    # Determine file type and load
    if file_path.endswith(".pt"):
        data = torch.load(file_path, weights_only=False)
    elif file_path.endswith(".npz"):
        npz_data = np.load(file_path)
        data = {key: npz_data[key] for key in npz_data.files}
    else:
        print(f"Unsupported file type: {file_path}")
        return

    print(f"\nLoaded: {file_path}")
    print(f"Keys in file: {list(data.keys())}")

    for cue_name, feat in data.items():
        # Convert to numpy if tensor
        if hasattr(feat, "numpy"):
            feat = feat.numpy()

        print(f"{cue_name} shape: {feat.shape}")

        # Plot first channel (if multiple channels)
        if feat.ndim == 3:  # [channels, time, freq]
            plt.imshow(np.abs(feat[0]), aspect='auto', origin='lower')
        elif feat.ndim == 2:  # [time, freq] or [freq, time]
            plt.imshow(feat, aspect='auto', origin='lower')
        else:
            print(f"Cannot plot {cue_name}, unsupported shape")
            continue

        plt.title(f"{cue_name} - {os.path.basename(file_path)}")
        plt.xlabel("Time frames")
        plt.ylabel("Frequency bins")
        plt.colorbar()
        plt.show()


def main():
    feature_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith((".pt", ".npz"))]
    feature_files.sort()

    for feature_file in feature_files:
        visualize_feature_file(os.path.join(FEATURES_DIR, feature_file))


if __name__ == "__main__":
    main()
