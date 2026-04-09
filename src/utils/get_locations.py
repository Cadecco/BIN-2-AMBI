# Extract spatial locations from .jams file in the dataset

import argparse
import csv
import json
import math
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"

DEFAULT_START_SCENE = 5400
DEFAULT_OUTPUT_FORMAT = "both"  # "csv", "json", or "both"


# ============================================================
# HELPERS
# ============================================================

def find_jams_file(scene_dir: Path) -> Path | None:
    """Find the .jams file inside a scene directory."""
    jams_files = list(scene_dir.glob("*.jams"))
    if len(jams_files) == 1:
        return jams_files[0]
    if len(jams_files) > 1:
        # Prefer the one matching the scene directory name
        expected = scene_dir / f"{scene_dir.name}.jams"
        if expected.exists():
            return expected
        return jams_files[0]
    return None


def parse_jams(jams_path: Path) -> list[dict]:
    """Parse a JAMS file and return a list of source position dicts."""
    with open(jams_path, "r") as f:
        data = json.load(f)

    sources = []
    for annotation in data.get("annotations", []):
        for obs in annotation.get("data", []):
            value = obs.get("value", {})
            role = value.get("role", "")
            if role != "foreground":
                continue

            az_rad = value.get("event_azimuth")
            el_rad = value.get("event_elevation")

            if az_rad is None or el_rad is None:
                continue

            az_deg = math.degrees(az_rad)
            el_deg = math.degrees(el_rad)

            sources.append({
                "event_id": value.get("event_id", ""),
                "azimuth_rad": az_rad,
                "elevation_rad": el_rad,
                "azimuth_deg": round(az_deg, 4),
                "elevation_deg": round(el_deg, 4),
                "snr": value.get("snr"),
                "event_time": value.get("event_time"),
                "event_duration": value.get("event_duration"),
            })

    return sources


def classify_azimuth_region(az_deg: float) -> str:
    """Classify azimuth into a region for front-back analysis.

    Normalises to [0, 360) then classifies:
      - front:      [-30, 30)   i.e. 330-360 or 0-30
      - front-side: [30, 60) or [300, 330)
      - side:       [60, 120) or [240, 300)
      - rear-side:  [120, 150) or [210, 240)
      - rear:       [150, 210)
    """
    az = az_deg % 360.0

    if az < 30 or az >= 330:
        return "front"
    if az < 60 or az >= 300:
        return "front-side"
    if az < 120 or az >= 240:
        return "side"
    if az < 150 or az >= 210:
        return "rear-side"
    return "rear"


def is_front_back_ambiguous(az_deg: float, threshold_deg: float = 30.0) -> bool:
    """Check if azimuth is near the median plane (front-back ambiguity zone).

    Returns True if the source is within threshold_deg of 0° or 180°.
    """
    az = az_deg % 360.0
    near_front = az < threshold_deg or az > (360 - threshold_deg)
    near_rear = abs(az - 180) < threshold_deg
    return near_front or near_rear


# ============================================================
# MAIN
# ============================================================

def extract_positions(dataset_name: str, start_scene: int, output_format: str):
    dataset_dir = DATASETS_DIR / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Available datasets: {[d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]}"
        )

    # Collect all scene directories at or above start_scene
    scene_dirs = []
    for d in sorted(dataset_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("scene"):
            continue
        try:
            scene_num = int(d.name.replace("scene", ""))
        except ValueError:
            continue
        if scene_num >= start_scene:
            scene_dirs.append(d)

    print(f"Found {len(scene_dirs)} scene directories (>= scene{start_scene:05d})")

    # Parse all JAMS files
    all_rows = []        # flat rows for CSV
    scene_data = {}      # nested dict for JSON

    n_found = 0
    n_missing = 0

    for scene_dir in sorted(scene_dirs):
        scene_id = scene_dir.name
        jams_path = find_jams_file(scene_dir)

        if jams_path is None:
            n_missing += 1
            continue

        sources = parse_jams(jams_path)
        n_found += 1

        scene_entry = {
            "scene_id": scene_id,
            "n_sources": len(sources),
            "sources": [],
        }

        has_fb_ambiguous = False

        for src in sources:
            region = classify_azimuth_region(src["azimuth_deg"])
            fb_ambiguous = is_front_back_ambiguous(src["azimuth_deg"])
            if fb_ambiguous:
                has_fb_ambiguous = True

            src_entry = {
                **src,
                "azimuth_region": region,
                "front_back_ambiguous": fb_ambiguous,
            }
            scene_entry["sources"].append(src_entry)

            all_rows.append({
                "scene_id": scene_id,
                "n_sources": len(sources),
                "event_id": src["event_id"],
                "azimuth_rad": src["azimuth_rad"],
                "elevation_rad": src["elevation_rad"],
                "azimuth_deg": src["azimuth_deg"],
                "elevation_deg": src["elevation_deg"],
                "snr": src["snr"],
                "event_time": src["event_time"],
                "event_duration": src["event_duration"],
                "azimuth_region": region,
                "front_back_ambiguous": fb_ambiguous,
            })

        scene_entry["has_fb_ambiguous_source"] = has_fb_ambiguous
        scene_data[scene_id] = scene_entry

    print(f"Parsed {n_found} JAMS files ({n_missing} missing)")
    print(f"Total source entries: {len(all_rows)}")

    # Summary stats
    n_fb_scenes = sum(1 for s in scene_data.values() if s["has_fb_ambiguous_source"])
    n_fb_sources = sum(1 for r in all_rows if r["front_back_ambiguous"])
    print(f"Scenes with front-back ambiguous source(s): {n_fb_scenes}/{n_found}")
    print(f"Individual sources in ambiguous zone: {n_fb_sources}/{len(all_rows)}")

    # Output
    output_dir = dataset_dir
    base_name = "test_scene_positions"

    if output_format in ("csv", "both"):
        csv_path = output_dir / f"{base_name}.csv"
        fieldnames = [
            "scene_id", "n_sources", "event_id",
            "azimuth_rad", "elevation_rad", "azimuth_deg", "elevation_deg",
            "snr", "event_time", "event_duration",
            "azimuth_region", "front_back_ambiguous",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV: {csv_path}")

    if output_format in ("json", "both"):
        json_path = output_dir / f"{base_name}.json"
        with open(json_path, "w") as f:
            json.dump(scene_data, f, indent=2)
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract source positions from JAMS files for test scenes."
    )
    parser.add_argument(
        "dataset_name",
        help="Name of the dataset folder inside datasets/ (e.g. 6000scenes_no_bg)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=DEFAULT_START_SCENE,
        help=f"First scene number to include (default: {DEFAULT_START_SCENE})",
    )
    parser.add_argument(
        "--output",
        choices=["csv", "json", "both"],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT})",
    )

    args = parser.parse_args()
    extract_positions(args.dataset_name, args.start, args.output)