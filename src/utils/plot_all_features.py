import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FEATURES_DIR = os.path.join(PROJECT_ROOT, "datasets", "6000scenes_no_bg", "features")
GT_DIR       = os.path.join(PROJECT_ROOT, "datasets", "6000scenes_no_bg", "groundtruth")

SCENE_FILTER = None  # Set to "scene1" etc. to filter specific scenes

# --- Display Options ---
SHOW_INPUT = True   # Set to False to hide input features
SHOW_GT = True       # Set to False to hide groundtruth

# --- Groundtruth Options ---
GT_START_WITH_DIFFUSENESS = True   # If showing GT only, put Diffuseness first
GT_SKIP_MEAN_MAG = True           # If True, skip GT channel 0 (mean magnitude)
GT_TRIANGLE_LAYOUT = True          # If showing GT only with Diffuseness,Y,Z,X -> put Diffuseness centered on top, Y/Z/X below

def find_scene_tag(filename: str):
    """Extracts 'sceneX' from filenames."""
    m = re.search(r"(scene\d+)", filename)
    return m.group(1) if m else None

def load_feature_file(file_path):
    """Loads .pt or .npz features into a dictionary of numpy arrays."""
    if file_path.endswith(".pt"):
        data = torch.load(file_path, weights_only=False)
    elif file_path.endswith(".npz"):
        npz_data = np.load(file_path)
        data = {key: npz_data[key] for key in npz_data.files}
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    out = {}
    for k, v in data.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        out[k] = v
    return out

def load_gt_file(gt_path):
    """Loads DirAC groundtruth (T, F, C)."""
    gt = np.load(gt_path)
    if gt.ndim != 3:
        raise ValueError(f"Expected GT shape (T, F, C), got {gt.shape}")
    return gt

def get_gt_channel_info(num_channels, show_gt_only=False):
    """
    Returns a list of (channel_index, channel_name) to plot.
    Groundtruth channel order:
      0: Mean Magnitude
      1: Diffuseness
      2: Y
      3: Z
      4: X
    """
    base_names = ["Mean Magnitude", "Diffuseness", "Y", "Z", "X"]

    channel_info = []
    for idx in range(num_channels):
        name = base_names[idx] if idx < len(base_names) else f"Channel {idx}"
        channel_info.append((idx, name))

    if GT_SKIP_MEAN_MAG:
        channel_info = [(idx, name) for idx, name in channel_info if idx != 0]

    if show_gt_only and GT_START_WITH_DIFFUSENESS:
        diffuseness = [(idx, name) for idx, name in channel_info if idx == 1]
        others = [(idx, name) for idx, name in channel_info if idx != 1]
        channel_info = diffuseness + others

    return channel_info

def use_gt_triangle_layout(gt_channels_to_plot, show_gt_only):
    """
    Use special layout only when GT-only mode and plotted channels are exactly:
    Diffuseness, Y, Z, X
    """
    names = [name for _, name in gt_channels_to_plot]
    return show_gt_only and GT_TRIANGLE_LAYOUT and names == ["Diffuseness", "Y", "Z", "X"]

def plot_scene(scene_tag, feat_path=None, gt_path=None):
    feats = load_feature_file(feat_path) if (SHOW_INPUT and feat_path) else {}
    gt = load_gt_file(gt_path) if (SHOW_GT and gt_path) else None

    feature_order = ["ILD", "IPD_sine", "IPD_cosine", "IC", "mean_mag", "rotation"]
    feat_keys = [k for k in feature_order if k in feats]

    show_gt_only = SHOW_GT and not SHOW_INPUT

    if gt is not None:
        _, _, C = gt.shape
        gt_channels_to_plot = get_gt_channel_info(C, show_gt_only=show_gt_only)
    else:
        gt_channels_to_plot = []

    special_gt_layout = use_gt_triangle_layout(gt_channels_to_plot, show_gt_only)

    # Determine grid dimensions
    nrows = 0
    ncols = 0

    if SHOW_INPUT and not SHOW_GT:
        nrows = 2
        ncols = 3
    elif special_gt_layout:
        nrows = 2
        ncols = 3
    else:
        if SHOW_INPUT:
            nrows += 1
            ncols = max(ncols, len(feat_keys))
        if SHOW_GT:
            nrows += 1
            ncols = max(ncols, len(gt_channels_to_plot))

    if nrows == 0 or ncols == 0:
        print("Warning: Nothing to display.")
        return

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    # Turn everything off first; enable only used axes
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c].axis("off")

    row = 0

    # Input Features
    if SHOW_INPUT:
        if not SHOW_GT:
            for idx, cue_name in enumerate(feat_keys):
                if idx < 3:
                    r, c = 0, idx
                else:
                    r, c = 1, idx - 3

                ax = axes[r, c]
                ax.axis("on")
                feat = feats[cue_name]

                is_rotation = cue_name.lower() == "rotation"

                if is_rotation and feat.ndim == 2:
                    img = feat
                    interpolation = "nearest"
                elif feat.ndim == 3:
                    img = np.abs(feat[0])
                    interpolation = "auto"
                else:
                    img = np.abs(feat)
                    interpolation = "auto"

                im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis", interpolation=interpolation)
                ax.set_title(f"Input: {cue_name}")
                ax.set_xlabel("Time frames")
                ax.set_ylabel("Freq bins" if not is_rotation else "Rotation levels")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            for col in range(min(len(feat_keys), ncols)):
                ax = axes[row, col]
                ax.axis("on")
                cue_name = feat_keys[col]
                feat = feats[cue_name]

                is_rotation = cue_name.lower() == "rotation"

                if is_rotation and feat.ndim == 2:
                    img = feat
                    interpolation = "nearest"
                elif feat.ndim == 3:
                    img = np.abs(feat[0])
                    interpolation = "auto"
                else:
                    img = np.abs(feat)
                    interpolation = "auto"

                im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis", interpolation=interpolation)
                ax.set_title(f"Input: {cue_name}")
                ax.set_xlabel("Time frames")
                ax.set_ylabel("Freq bins" if not is_rotation else "Rotation levels")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            row += 1

    # Ground Truth
    if SHOW_GT:
        if special_gt_layout:
            # Layout:
            #   [off] [Diffuseness] [off]
            #   [Y]   [Z]           [X]
            layout_positions = {
                0: (0, 1),  # Diffuseness
                1: (1, 0),  # Y
                2: (1, 1),  # Z
                3: (1, 2),  # X
            }

            for idx, (gt_idx, gt_name) in enumerate(gt_channels_to_plot):
                r, c = layout_positions[idx]
                ax = axes[r, c]
                ax.axis("on")

                img = gt[:, :, gt_idx].T
                im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")
                ax.set_title(f"GT: {gt_name}")
                ax.set_xlabel("Time frames")
                ax.set_ylabel("Freq bins")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            for col in range(len(gt_channels_to_plot)):
                ax = axes[row, col]
                ax.axis("on")

                gt_idx, gt_name = gt_channels_to_plot[col]
                img = gt[:, :, gt_idx].T
                im = ax.imshow(img, aspect="auto", origin="lower", cmap="viridis")

                ax.set_title(f"GT: {gt_name}")
                ax.set_xlabel("Time frames")
                ax.set_ylabel("Freq bins")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    feat_name = os.path.basename(feat_path) if feat_path else "N/A"
    gt_name = os.path.basename(gt_path) if gt_path else "N/A"

    fig.suptitle(f"Scene: {scene_tag}\nFeatures: {feat_name} | GT: {gt_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    feat_files = sorted([f for f in os.listdir(FEATURES_DIR) if f.endswith((".pt", ".npz"))])
    gt_files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".npy")])

    feat_map = {find_scene_tag(f): os.path.join(FEATURES_DIR, f) for f in feat_files if find_scene_tag(f)}
    gt_map = {find_scene_tag(f): os.path.join(GT_DIR, f) for f in gt_files if find_scene_tag(f)}

    if SHOW_INPUT and SHOW_GT:
        scenes = sorted(set(feat_map.keys()) & set(gt_map.keys()))
    elif SHOW_INPUT:
        scenes = sorted(feat_map.keys())
    elif SHOW_GT:
        scenes = sorted(gt_map.keys())
    else:
        scenes = []

    if SCENE_FILTER:
        scenes = [s for s in scenes if s == SCENE_FILTER]

    if not scenes:
        print("No matching scenes found.")
        return

    for scene_tag in scenes:
        feat_path = feat_map.get(scene_tag)
        gt_path = gt_map.get(scene_tag)

        if SHOW_INPUT and not feat_path:
            print(f"[Warning] Missing features for {scene_tag}")
        if SHOW_GT and not gt_path:
            print(f"[Warning] Missing GT for {scene_tag}")

        plot_scene(scene_tag, feat_path, gt_path)

if __name__ == "__main__":
    main()