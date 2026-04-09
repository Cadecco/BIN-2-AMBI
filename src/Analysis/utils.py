# Utils file from Nils Peters:

import numpy as np
import random
from pathlib import Path
import json
import soundfile as sf
import librosa
from scipy.fft import fft, ifft
from scipy import signal

import os
import tempfile
import re

# Split files into train, val and test sets for data generatin to avoid leakage.
# Shuffle seed is deteministic so that generation may be resumed without changing the splits.
def split_files_unique(files, train_p=0.8, val_p=0.1, test_p=0.1, seed=42):
    assert abs((train_p + val_p + test_p) - 1.0) < 1e-9
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_p)
    n_val = int(n * val_p)

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:]
    }


# Change "w" to "a" for APPEND mode
def append_to_manifest(row, dataset_path):
    manifest_path = Path(dataset_path) / "manifest.jsonl"
    
    # "a" ensures we add to the bottom, not delete the top
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_existing_ids(manifest_path):
    if not manifest_path.exists():
        return set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        # Get the scene_id from every line already in the file
        return {json.loads(line)["scene_id"] for line in f}
    
# Extend the duration of an event if the scene will be too sparse
def stitch_to_duration(chosen_file, samples_folder, file_pool,
                        target_duration, sample_rate,
                        min_fill=0.90, min_gap=0.3, max_gap=1.0,
                        crossfade_duration=0.05):
    """
    Stitch audio files from the same class folder until min_fill * target_duration is reached.
    Files are separated by a random silence gap with a short crossfade to avoid clicks.
    Returns (temp_file_path, list_of_files_used).
    """
    target_samples = int(target_duration * sample_rate)
    min_samples = int(target_samples * min_fill)
    crossfade_samples = int(crossfade_duration * sample_rate)

    source_folder = os.path.dirname(chosen_file)
    folder_files = [f for f in file_pool if os.path.dirname(f) == source_folder]
    if not folder_files:
        folder_files = file_pool

    stitched_files = [chosen_file]

    def load_and_resample(filepath):
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != sample_rate:
            audio = resampy.resample(audio, sr, sample_rate)
        return audio

    def apply_crossfade(chunk):
        """Fade out at end of chunk."""
        if len(chunk) < crossfade_samples:
            return chunk
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        chunk = chunk.copy()
        chunk[-crossfade_samples:] *= fade_out
        return chunk

    # Start with the originally chosen file
    audio = load_and_resample(os.path.join(samples_folder, chosen_file))
    audio = apply_crossfade(audio)
    
    chunks = [audio]
    total_samples = len(audio)

    while total_samples < min_samples:
        # Silent gap
        gap_samples = int(random.uniform(min_gap, max_gap) * sample_rate)
        chunks.append(np.zeros(gap_samples))
        total_samples += gap_samples

        # Next file with crossfade applied
        next_file = random.choice(folder_files)
        audio = load_and_resample(os.path.join(samples_folder, next_file))
        audio = apply_crossfade(audio)
        chunks.append(audio)
        total_samples += len(audio)
        stitched_files.append(next_file)

    stitched = np.concatenate(chunks)[:target_samples]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, stitched, sample_rate)

    return tmp.name, stitched_files

# Get headtracking speed for synthetic head tracking data.
HT_SPEED_RE = re.compile(r"^ht-(\d+)-[a-z]\.flac$", re.IGNORECASE)

def parse_ht_speed_deg_per_sec(filename: str):
    m = HT_SPEED_RE.match(filename)
    if not m:
        return None
    return float(m.group(1))