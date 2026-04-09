import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sys
import os

def plot_yaw(flac_file):

    plt.rcParams.update({
    "font.size": 10,         # base (matches LaTeX body)
    "axes.labelsize": 10,    # axis labels
    "xtick.labelsize": 8,   # ticks slightly smaller
    "ytick.labelsize": 8,
    "axes.titlesize": 11,    # slightly bigger
})
    
    # Load audio file
    data, samplerate = sf.read(flac_file)

    # Take first channel if multi-channel
    if len(data.shape) > 1:
        channel_1 = data[:, 0]
    else:
        channel_1 = data

    # Convert [-1, 1] → [-180, 180] degrees
    yaw_degrees = channel_1 * 180

    # Time axis (starts exactly at 0, ends at last sample)
    time = np.arange(len(channel_1)) / samplerate

    # Plot
    plt.figure(figsize=(6, 2))
    plt.plot(time, yaw_degrees, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")

    filename = os.path.basename(flac_file)
    plt.title(f"Yaw Rotation for {filename}")

    # ✅ FIXED AXES
    plt.ylim(-180, 180)           # always full yaw range
    plt.xlim(0, time[-1])         # no padding, exact duration

    # add horizontal middle line
    plt.axhline(0, linestyle='--', linewidth=1, color='0.5')

    plt.grid(False)
    plt.tight_layout()


def main():
    if len(sys.argv) < 2:
        print("Drag and drop one or more .flac files onto this script or pass them as arguments.")
        sys.exit(1)

    files = sys.argv[1:]

    for f in files:
        plot_yaw(f)

    plt.show()


if __name__ == "__main__":
    main()