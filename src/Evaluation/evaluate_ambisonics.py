# Run evaluation from AMBIQUAL

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm


@dataclass
class SceneRow:
    scene_id: str
    gt_path: str
    pred_path: str
    GT_Resynth_LQ: Optional[float]
    GT_Resynth_LA: Optional[float]
    Pred_Resynth_LQ: Optional[float]
    Pred_Resynth_LA: Optional[float]
    LQ_percent: Optional[float]
    LA_percent: Optional[float]


def _find_audio_files(root: Path, exts: Set[str]) -> List[Path]:
    """
    Recursively find audio files under root that match extensions in `exts`.
    Extensions are compared case-insensitively and should include the dot, e.g. {".wav", ".flac"}.
    """
    exts_lower = {e.lower() for e in exts}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_lower]


def _index_by_scene(folder: Path, scene_re: re.Pattern, exts: Set[str]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for f in _find_audio_files(folder, exts):
        m = scene_re.search(f.name)
        if not m:
            continue
        sid = m.group(1).lower()
        if sid in idx:
            raise ValueError(
                f"Multiple files found for scene id '{sid}' in {folder}:\n"
                f"  - {idx[sid]}\n"
                f"  - {f}\n"
                f"Expected exactly one per scene."
            )
        idx[sid] = f
    return idx


def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    k = (n - 1) * p
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None, "p10": None, "p90": None}

    s = sorted(vals)
    n = len(s)
    mean = statistics.mean(s)
    std = statistics.stdev(s) if n >= 2 else 0.0

    return {
        "mean": float(mean),
        "std": float(std),
        "min": float(s[0]),
        "max": float(s[-1]),
        "median": float(statistics.median(s)),
        "p10": float(_percentile(s, 0.10)),
        "p90": float(_percentile(s, 0.90)),
    }


def evaluate_dataset(
    *,
    pred_resynth_dir: Path,
    gt_dir: Path,
    gt_resynth_dir: Path,
    out_csv: Path,
    scene_regex: str,
    calculate_ambiqual_fn,
    intensity_threshold: int,
    elc: int,
    ignore_freq_bands: int,
    # allow different extensions 
    gt_exts: Set[str] = {".wav", ".flac"},
    resynth_exts: Set[str] = {".wav", ".flac"},
) -> Dict[str, object]:
    """
    Evaluates both GT resynth and prediction resynth against the original FOA.
    Then compares pred_resynth scores as a percentage of gt_resynth scores.
    """
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")
    if not gt_resynth_dir.exists():
        raise FileNotFoundError(f"gt_resynth_dir not found: {gt_resynth_dir}")
    if not pred_resynth_dir.exists():
        raise FileNotFoundError(f"pred_resynth_dir not found: {pred_resynth_dir}")

    scene_re = re.compile(scene_regex, flags=re.IGNORECASE)

    gt_idx = _index_by_scene(gt_dir, scene_re, gt_exts)
    gt_resynth_idx = _index_by_scene(gt_resynth_dir, scene_re, resynth_exts)
    pred_resynth_idx = _index_by_scene(pred_resynth_dir, scene_re, resynth_exts)

    common = sorted(set(gt_idx) & set(gt_resynth_idx) & set(pred_resynth_idx))
    missing_pred = sorted(set(gt_idx) - set(pred_resynth_idx))
    missing_gt_resynth = sorted(set(gt_idx) - set(gt_resynth_idx))

    if not common:
        raise RuntimeError(
            "No matching scenes across all three sources (GT, GT resynth, Pred resynth).\n"
            f"GT indexed: {len(gt_idx)} | GT resynth indexed: {len(gt_resynth_idx)} | Pred resynth indexed: {len(pred_resynth_idx)}\n"
            f"Scene regex: {scene_regex}\n"
            f"GT extensions: {sorted(gt_exts)} | Resynth extensions: {sorted(resynth_exts)}\n"
            "Check filenames (expected e.g. scene12_...)."
        )

    rows: List[SceneRow] = []
    for sid in tqdm(common, desc="Evaluating scenes"):
        gt_path = gt_idx[sid]
        gt_resynth_path = gt_resynth_idx[sid]
        pred_resynth_path = pred_resynth_idx[sid]

        # Evaluate GT resynth vs original FOA
        _, gt_resynth_lq, gt_resynth_la = calculate_ambiqual_fn(
            ref_path=gt_path,
            deg_path=gt_resynth_path,
            intensity_threshold=intensity_threshold,
            elc=elc,
            ignore_freq_bands=ignore_freq_bands,
        )

        # Evaluate Pred resynth vs original FOA
        _, pred_resynth_lq, pred_resynth_la = calculate_ambiqual_fn(
            ref_path=gt_path,
            deg_path=pred_resynth_path,
            intensity_threshold=intensity_threshold,
            elc=elc,
            ignore_freq_bands=ignore_freq_bands,
        )

        # Calculate percentages: (pred_resynth / gt_resynth) * 100
        lq_percent = None
        la_percent = None
        if gt_resynth_lq is not None and gt_resynth_lq != 0 and pred_resynth_lq is not None:
            lq_percent = (float(pred_resynth_lq) / float(gt_resynth_lq)) * 100.0
        if gt_resynth_la is not None and gt_resynth_la != 0 and pred_resynth_la is not None:
            la_percent = (float(pred_resynth_la) / float(gt_resynth_la)) * 100.0

        rows.append(
            SceneRow(
                scene_id=sid,
                gt_path=str(gt_path),
                pred_path=str(pred_resynth_path),
                GT_Resynth_LQ=float(gt_resynth_lq) if gt_resynth_lq is not None else None,
                GT_Resynth_LA=float(gt_resynth_la) if gt_resynth_la is not None else None,
                Pred_Resynth_LQ=float(pred_resynth_lq) if pred_resynth_lq is not None else None,
                Pred_Resynth_LA=float(pred_resynth_la) if pred_resynth_la is not None else None,
                LQ_percent=lq_percent,
                LA_percent=la_percent,
            )
        )

    # Write per-scene CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id", "GT_Resynth_LQ", "GT_Resynth_LA", "Pred_Resynth_LQ", "Pred_Resynth_LA", "LQ_percent", "LA_percent", "gt_path", "pred_path"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "scene_id": r.scene_id,
                    "GT_Resynth_LQ": r.GT_Resynth_LQ,
                    "GT_Resynth_LA": r.GT_Resynth_LA,
                    "Pred_Resynth_LQ": r.Pred_Resynth_LQ,
                    "Pred_Resynth_LA": r.Pred_Resynth_LA,
                    "LQ_percent": r.LQ_percent,
                    "LA_percent": r.LA_percent,
                    "gt_path": r.gt_path,
                    "pred_path": r.pred_path,
                }
            )

    gt_resynth_lq_vals = [r.GT_Resynth_LQ for r in rows if r.GT_Resynth_LQ is not None]
    gt_resynth_la_vals = [r.GT_Resynth_LA for r in rows if r.GT_Resynth_LA is not None]
    pred_resynth_lq_vals = [r.Pred_Resynth_LQ for r in rows if r.Pred_Resynth_LQ is not None]
    pred_resynth_la_vals = [r.Pred_Resynth_LA for r in rows if r.Pred_Resynth_LA is not None]
    lq_percent_vals = [r.LQ_percent for r in rows if r.LQ_percent is not None]
    la_percent_vals = [r.LA_percent for r in rows if r.LA_percent is not None]

    return {
        "n_scored": len(rows),
        "n_gt_indexed": len(gt_idx),
        "n_gt_resynth_indexed": len(gt_resynth_idx),
        "n_pred_resynth_indexed": len(pred_resynth_idx),
        "missing_pred_resynth": missing_pred,
        "missing_gt_resynth": missing_gt_resynth,
        "GT_Resynth_LQ": _stats([float(x) for x in gt_resynth_lq_vals]),
        "GT_Resynth_LA": _stats([float(x) for x in gt_resynth_la_vals]),
        "Pred_Resynth_LQ": _stats([float(x) for x in pred_resynth_lq_vals]),
        "Pred_Resynth_LA": _stats([float(x) for x in pred_resynth_la_vals]),
        "LQ_percent": _stats([float(x) for x in lq_percent_vals]),
        "LA_percent": _stats([float(x) for x in la_percent_vals]),
    }
