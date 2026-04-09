import numpy as np
import soundfile as sf
import os
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HT_FOLDER = PROJECT_ROOT / "resources" / "ht-synthetic/"

# ==========================
# Configuration
# ==========================
output_dir = HT_FOLDER
os.makedirs(output_dir, exist_ok=True)

duration_sec = 10.0
sample_rate = 48000

speeds_deg_per_sec = [10, 20, 50, 100, 200, 400]

pause_duration_sec = 0.5
pause_samples = int(pause_duration_sec * sample_rate)

LOW = -1.0
HIGH = 1.0

total_samples = int(duration_sec * sample_rate)

# Variants:
# a = 0°
# b = -180°
# c = +180°
start_positions = {
    "a": 0.0,
    "b": 1.0,
    "c": -1.0
}

for speed_deg in speeds_deg_per_sec:

    speed_norm_per_sec = speed_deg / 180.0
    speed_norm_per_sample = speed_norm_per_sec / sample_rate

    for label, start_val in start_positions.items():

        yaw = np.zeros(total_samples, dtype=np.float32)

        current = start_val

        # Direction depends on start point
        if start_val <= LOW:
            direction = 1
        elif start_val >= HIGH:
            direction = -1
        else:
            direction = 1

        pause_counter = 0

        for i in range(total_samples):

            if pause_counter > 0:
                yaw[i] = current
                pause_counter -= 1
                continue

            next_val = current + direction * speed_norm_per_sample

            if next_val >= HIGH:
                current = HIGH
                yaw[i] = current
                direction = -1
                pause_counter = pause_samples
                continue

            if next_val <= LOW:
                current = LOW
                yaw[i] = current
                direction = 1
                pause_counter = pause_samples
                continue

            current = next_val
            yaw[i] = current

        pitch = np.zeros_like(yaw)
        roll = np.zeros_like(yaw)

        ht = np.stack([yaw, pitch, roll], axis=1)

        filename = os.path.join(
            output_dir,
            f"ht-{speed_deg}-{label}.flac"
        )

        sf.write(filename, ht, sample_rate)

        print(f"Saved: {filename}")