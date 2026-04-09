from __future__ import annotations

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# MATLAB default color order
MATLAB_COLORS = {
    "blue": "#0072BD",
    "orange": "#D95319",
    "yellow": "#EDB120",
    "purple": "#7E2F8E",
    "green": "#77AC30",
    "cyan": "#4DBEEE",
    "darkred": "#A2142F",
}


TITLE_FONTSIZE = 11
LABEL_FONTSIZE = 10
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 9


def load_manifest(manifest_path: Path) -> Dict[str, int]:
    """Load manifest.jsonl and extract scene_id -> n_sources mapping."""
    scene_sources = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            scene_id = data["scene_id"]
            n_sources = data["n_sources"]
            scene_sources[scene_id] = n_sources
    return scene_sources


def load_ambiqual_csv(
    csv_path: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load ambiqual CSV and extract scene_id -> LA mappings.

    Returns:
        Tuple of (scene_gt_la, scene_pred_la, scene_la_percent)
    """
    scene_gt_la = {}
    scene_pred_la = {}
    scene_la_percent = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row["scene_id"]

            gt_la = row.get("GT_Resynth_LA")
            if gt_la and gt_la.lower() != "none":
                try:
                    scene_gt_la[scene_id] = float(gt_la)
                except ValueError:
                    pass

            pred_la = row.get("Pred_Resynth_LA")
            if pred_la and pred_la.lower() != "none":
                try:
                    scene_pred_la[scene_id] = float(pred_la)
                except ValueError:
                    pass

            la_percent = row.get("LA_percent")
            if la_percent and la_percent.lower() != "none":
                try:
                    scene_la_percent[scene_id] = float(la_percent)
                except ValueError:
                    pass

    return scene_gt_la, scene_pred_la, scene_la_percent


def merge_single_experiment_data(
    scene_sources: Dict[str, int],
    scene_gt_la: Dict[str, float],
    scene_pred_la: Dict[str, float],
    scene_la_percent: Dict[str, float],
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Merge manifest and one ambiqual CSV, keeping only scenes present in all inputs."""
    n_sources_list = []
    gt_la_list = []
    pred_la_list = []
    la_percent_list = []

    for scene_id, gt_la in scene_gt_la.items():
        if (
            scene_id in scene_sources
            and scene_id in scene_pred_la
            and scene_id in scene_la_percent
        ):
            n_sources_list.append(scene_sources[scene_id])
            gt_la_list.append(gt_la)
            pred_la_list.append(scene_pred_la[scene_id])
            la_percent_list.append(scene_la_percent[scene_id])

    return n_sources_list, gt_la_list, pred_la_list, la_percent_list


def merge_comparison_data(
    scene_sources: Dict[str, int],
    mse_gt_la: Dict[str, float],
    mse_pred_la: Dict[str, float],
    mse_la_percent: Dict[str, float],
    mae_gt_la: Dict[str, float],
    mae_pred_la: Dict[str, float],
    mae_la_percent: Dict[str, float],
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Merge manifest and both ambiqual CSVs, keeping only scenes present in all inputs."""
    n_sources_list = []
    gt_la_list = []
    mse_pred_la_list = []
    mse_la_percent_list = []
    mae_pred_la_list = []
    mae_la_percent_list = []

    common_scene_ids = (
        set(scene_sources.keys())
        & set(mse_gt_la.keys())
        & set(mse_pred_la.keys())
        & set(mse_la_percent.keys())
        & set(mae_gt_la.keys())
        & set(mae_pred_la.keys())
        & set(mae_la_percent.keys())
    )

    for scene_id in sorted(common_scene_ids):
        n_sources_list.append(scene_sources[scene_id])
        # GT should be the same in both; use MSE copy
        gt_la_list.append(mse_gt_la[scene_id])
        mse_pred_la_list.append(mse_pred_la[scene_id])
        mse_la_percent_list.append(mse_la_percent[scene_id])
        mae_pred_la_list.append(mae_pred_la[scene_id])
        mae_la_percent_list.append(mae_la_percent[scene_id])

    return (
        n_sources_list,
        gt_la_list,
        mse_pred_la_list,
        mse_la_percent_list,
        mae_pred_la_list,
        mae_la_percent_list,
    )


def compute_group_stats(
    n_sources_list: List[int], values: List[float]
) -> Tuple[List[int], List[float], List[float], List[int]]:
    """Compute mean/std/count for each unique number of sources."""
    unique_sources = sorted(set(n_sources_list))
    means = []
    stds = []
    counts = []

    for n_src in unique_sources:
        vals = [v for n, v in zip(n_sources_list, values) if n == n_src]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        counts.append(len(vals))

    return unique_sources, means, stds, counts


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def experiment_title_suffix(experiment_name: str) -> str:
    name = experiment_name.lower()
    labels = []

    if "no_bg" in name or "no-background" in name:
        labels.append("No Background")
    elif "bg" in name or "background" in name:
        labels.append("W/Background")

    if "synthetic_ht" in name:
        labels.append("Synthetic HT")

    if not labels:
        return ""

    return f" ({', '.join(labels)})"


def plot_single_experiment(
    n_sources_list: List[int],
    gt_la_list: List[float],
    pred_la_list: List[float],
    la_percent_list: List[float],
    output_path: Path,
    pred_label: str = "Pred (MSE)",
    title_suffix: str = "",
) -> None:
    """Create plots for a single experiment."""
    unique_sources, gt_means, gt_stds, counts = compute_group_stats(n_sources_list, gt_la_list)
    _, pred_means, pred_stds, _ = compute_group_stats(n_sources_list, pred_la_list)
    _, percent_means, percent_stds, _ = compute_group_stats(n_sources_list, la_percent_list)

    # ===== Plot 1: Absolute LA =====
    fig1, ax1 = plt.subplots(figsize=(6.3, 4.2))  # A4-friendly

    ax1.errorbar(
        unique_sources,
        gt_means,
        yerr=gt_stds,
        fmt="o-",
        color="tab:blue",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="GT",
        elinewidth=1.5,
    )
    ax1.errorbar(
        unique_sources,
        pred_means,
        yerr=pred_stds,
        fmt="s-",
        color="tab:orange",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label=pred_label,
        elinewidth=1.5,
    )

    ax1.set_xlabel("Number of Events in Scene", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("LA (Absolute Score 0–1)", fontsize=LABEL_FONTSIZE)
    ax1.set_title(f"Absolute LA vs Number of Events{title_suffix}", fontsize=TITLE_FONTSIZE)
    ax1.set_xticks(unique_sources)
    ax1.set_ylim([0, 1])
    style_axes(ax1)
    ax1.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    plt.tight_layout()
    output_absolute = output_path.with_stem(output_path.stem + "_absolute")
    plt.savefig(output_absolute, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {output_absolute}")
    plt.close(fig1)

    # ===== Plot 2: Relative LA =====
    fig2, ax2 = plt.subplots(figsize=(6.3, 4.2))

    ax2.scatter(
        n_sources_list,
        la_percent_list,
        alpha=0.25,
        s=20,
        edgecolors="tab:blue",
        linewidth=0.5,
        facecolors="none",
    )
    ax2.errorbar(
        unique_sources,
        percent_means,
        yerr=percent_stds,
        fmt="o-",
        color="tab:blue",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label=f"{pred_label} Mean ± Std",
        elinewidth=1.5,
    )
    ax2.axhline(
        y=100,
        color=MATLAB_COLORS["darkred"],
        linestyle="--",
        linewidth=1.5,
        label="Perfect (100%)",
    )

    ax2.set_xlabel("Number of Events in Scene", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("LA (Pred as % of GT)", fontsize=LABEL_FONTSIZE)
    ax2.set_title(f"Relative LA vs Number of Events{title_suffix}", fontsize=TITLE_FONTSIZE)
    ax2.set_xticks(unique_sources)
    style_axes(ax2)
    ax2.legend(fontsize=LEGEND_FONTSIZE, loc="lower right")

    plt.tight_layout()
    ax2.set_ylim([40, 120])
    output_relative = output_path.with_stem(output_path.stem + "_relative")
    plt.savefig(output_relative, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {output_relative}")
    plt.close(fig2)

    print("\n=== LA Statistics by Number of Sources ===")
    for n_src, gt_mean, gt_std, pred_mean, pred_std, pct_mean, pct_std, count in zip(
        unique_sources, gt_means, gt_stds, pred_means, pred_stds, percent_means, percent_stds, counts
    ):
        print(f"Sources: {n_src} | Count: {count:4d}")
        print(f"  GT:         Mean: {gt_mean:6.4f} | Std: {gt_std:6.4f}")
        print(f"  {pred_label}: Mean: {pred_mean:6.4f} | Std: {pred_std:6.4f}")
        print(f"  % of GT:    Mean: {pct_mean:6.2f}% | Std: {pct_std:6.2f}%")


def plot_experiment_comparison(
    n_sources_list: List[int],
    gt_la_list: List[float],
    mse_pred_la_list: List[float],
    mse_la_percent_list: List[float],
    mae_pred_la_list: List[float],
    mae_la_percent_list: List[float],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create plots comparing MSE and MAE experiments."""
    unique_sources, gt_means, gt_stds, counts = compute_group_stats(n_sources_list, gt_la_list)
    _, mse_pred_means, mse_pred_stds, _ = compute_group_stats(n_sources_list, mse_pred_la_list)
    _, mae_pred_means, mae_pred_stds, _ = compute_group_stats(n_sources_list, mae_pred_la_list)
    _, mse_percent_means, mse_percent_stds, _ = compute_group_stats(n_sources_list, mse_la_percent_list)
    _, mae_percent_means, mae_percent_stds, _ = compute_group_stats(n_sources_list, mae_la_percent_list)

    # ===== Plot 1: Absolute LA =====
    fig1, ax1 = plt.subplots(figsize=(6.8, 4.4))

    ax1.errorbar(
        unique_sources,
        gt_means,
        yerr=gt_stds,
        fmt="o-",
        color="tab:blue",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="GT",
        elinewidth=1.5,
    )
    ax1.errorbar(
        unique_sources,
        mse_pred_means,
        yerr=mse_pred_stds,
        fmt="s-",
        color="tab:orange",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="Pred (MSE)",
        elinewidth=1.5,
    )
    ax1.errorbar(
        unique_sources,
        mae_pred_means,
        yerr=mae_pred_stds,
        fmt="^-",
        color="tab:green",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="Pred (MAE)",
        elinewidth=1.5,
    )

    ax1.set_xlabel("Number of Events in Scene", fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel("LA (Absolute Score 0–1)", fontsize=LABEL_FONTSIZE)
    ax1.set_title(f"Absolute LA vs Number of Events{title_suffix}", fontsize=TITLE_FONTSIZE)
    ax1.set_xticks(unique_sources)
    ax1.set_ylim([0, 1])
    style_axes(ax1)
    ax1.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

    plt.tight_layout()
    output_absolute = output_path.with_stem(output_path.stem + "_absolute")
    plt.savefig(output_absolute, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {output_absolute}")
    plt.close(fig1)

    # ===== Plot 2: Relative LA =====
    fig2, ax2 = plt.subplots(figsize=(6.8, 4.4))

    ax2.errorbar(
        unique_sources,
        mse_percent_means,
        yerr=mse_percent_stds,
        fmt="s-",
        color="tab:blue",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="Pred (MSE) as % of GT",
        elinewidth=1.5,
    )
    ax2.errorbar(
        unique_sources,
        mae_percent_means,
        yerr=mae_percent_stds,
        fmt="^-",
        color="tab:green",
        linewidth=2.0,
        markersize=5,
        capsize=4,
        label="Pred (MAE) as % of GT",
        elinewidth=1.5,
    )
    ax2.axhline(
        y=100,
        color=MATLAB_COLORS["darkred"],
        linestyle="--",
        linewidth=1.5,
        label="Perfect (100%)",
    )

    ax2.set_xlabel("Number of Events in Scene", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("LA (Pred as % of GT)", fontsize=LABEL_FONTSIZE)
    ax2.set_title(f"Relative LA vs Number of Events{title_suffix}", fontsize=TITLE_FONTSIZE)
    ax2.set_xticks(unique_sources)
    style_axes(ax2)
    ax2.legend(fontsize=LEGEND_FONTSIZE, loc="lower right")

    plt.tight_layout()
    ax2.set_ylim([40, 120])
    output_relative = output_path.with_stem(output_path.stem + "_relative")
    plt.savefig(output_relative, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {output_relative}")
    plt.close(fig2)

    print("\n=== LA Statistics by Number of Sources (MSE vs MAE) ===")
    for (
        n_src,
        gt_mean,
        gt_std,
        mse_mean,
        mse_std,
        mae_mean,
        mae_std,
        mse_pct_mean,
        mse_pct_std,
        mae_pct_mean,
        mae_pct_std,
        count,
    ) in zip(
        unique_sources,
        gt_means,
        gt_stds,
        mse_pred_means,
        mse_pred_stds,
        mae_pred_means,
        mae_pred_stds,
        mse_percent_means,
        mse_percent_stds,
        mae_percent_means,
        mae_percent_stds,
        counts,
    ):
        print(f"Sources: {n_src} | Count: {count:4d}")
        print(f"  GT:         Mean: {gt_mean:6.4f} | Std: {gt_std:6.4f}")
        print(f"  Pred (MSE): Mean: {mse_mean:6.4f} | Std: {mse_std:6.4f}")
        print(f"  Pred (MAE): Mean: {mae_mean:6.4f} | Std: {mae_std:6.4f}")
        print(f"  MSE % GT:   Mean: {mse_pct_mean:6.2f}% | Std: {mse_pct_std:6.2f}%")
        print(f"  MAE % GT:   Mean: {mae_pct_mean:6.2f}% | Std: {mae_pct_std:6.2f}%")


def main() -> None:
    # -------------------------
    # CONFIG
    # -------------------------
    ROOT = Path(__file__).resolve().parents[2]
    DATASET_NAME = "6000scenes"

    # Base experiment stem. Folders expected:
    #   <BASE_EXPERIMENT_NAME>_MSE
    #   <BASE_EXPERIMENT_NAME>_MAE
    BASE_EXPERIMENT_NAME = "CRNN_V15_Rotation_BG"

    # Toggle comparison mode
    INCLUDE_MAE_COMPARISON = True

    # If False, only this experiment is used
    SINGLE_EXPERIMENT_NAME = f"{BASE_EXPERIMENT_NAME}"

    # Derived comparison experiment names
    MSE_EXPERIMENT_NAME = f"{BASE_EXPERIMENT_NAME}"
    MAE_EXPERIMENT_NAME = f"{BASE_EXPERIMENT_NAME}_MAE"

    MANIFEST_PATH = ROOT / "Experiments" / MSE_EXPERIMENT_NAME / "manifest.jsonl"

    title_suffix = experiment_title_suffix(MSE_EXPERIMENT_NAME)

    # -------------------------
    # Load manifest
    # -------------------------
    print(f"Loading manifest from: {MANIFEST_PATH}")
    scene_sources = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(scene_sources)} scenes from manifest")

    if INCLUDE_MAE_COMPARISON:
        mse_csv = ROOT / "Experiments" / MSE_EXPERIMENT_NAME / "ambiqual_per_scene.csv"
        mae_csv = ROOT / "Experiments" / MAE_EXPERIMENT_NAME / "ambiqual_per_scene.csv"
        output_plot = ROOT / "Experiments" / MSE_EXPERIMENT_NAME / "la_vs_sources_mse_vs_mae.pdf"

        print(f"Loading MSE ambiqual CSV from: {mse_csv}")
        mse_gt_la, mse_pred_la, mse_la_percent = load_ambiqual_csv(mse_csv)

        print(f"Loading MAE ambiqual CSV from: {mae_csv}")
        mae_gt_la, mae_pred_la, mae_la_percent = load_ambiqual_csv(mae_csv)

        (
            n_sources_list,
            gt_la_list,
            mse_pred_la_list,
            mse_la_percent_list,
            mae_pred_la_list,
            mae_la_percent_list,
        ) = merge_comparison_data(
            scene_sources,
            mse_gt_la,
            mse_pred_la,
            mse_la_percent,
            mae_gt_la,
            mae_pred_la,
            mae_la_percent,
        )

        print(f"Merged {len(n_sources_list)} scenes with data in all files")

        output_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_experiment_comparison(
            n_sources_list,
            gt_la_list,
            mse_pred_la_list,
            mse_la_percent_list,
            mae_pred_la_list,
            mae_la_percent_list,
            output_plot,
            title_suffix=title_suffix,
        )

    else:
        ambiqual_csv = ROOT / "Experiments" / SINGLE_EXPERIMENT_NAME / "ambiqual_per_scene.csv"
        output_plot = ROOT / "Experiments" / SINGLE_EXPERIMENT_NAME / "la_vs_sources.pdf"

        print(f"Loading ambiqual CSV from: {ambiqual_csv}")
        scene_gt_la, scene_pred_la, scene_la_percent = load_ambiqual_csv(ambiqual_csv)

        n_sources_list, gt_la_list, pred_la_list, la_percent_list = merge_single_experiment_data(
            scene_sources, scene_gt_la, scene_pred_la, scene_la_percent
        )
        print(f"Merged {len(n_sources_list)} scenes with data in all files")

        output_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_single_experiment(
            n_sources_list,
            gt_la_list,
            pred_la_list,
            la_percent_list,
            output_plot,
            pred_label="Pred (MSE)",
            title_suffix=title_suffix,
        )


if __name__ == "__main__":
    main()