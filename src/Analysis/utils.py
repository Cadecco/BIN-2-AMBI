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
import resampy
import re

def get_Audio(Audio_filename, idx, fsFramework):

    with open(Audio_filename, 'rb') as f:
        info = sf.info(Audio_filename)
        print(f"Sampling Rate: {info.samplerate} Hz")
        insig, fs = sf.read(f)
        insig = insig[:,idx]
        assert(fs==fsFramework)
        #self._fs=fs
        #self.lInsig = np.shape(insig)[0]
        #self.nInChan = np.shape(insig)[1]

    return insig

def get_spectrogram(config, insig_):

    overlap = config["analysis"]["overlap"]
    winsize = config["analysis"]["window_size"]
    start_freq = config["analysis"]["start_freq"]
    stop_freq = config["analysis"]["stop_freq"]
    # Use round to match up with binaspect implementation
    hop_length = round(winsize * overlap) # Was np.ceil
    bin_width = config["analysis"]["sample_rate"] / winsize 

    lInsig = np.shape(insig_)[0]
    nInChan = np.shape(insig_)[1]
    Nhop = round(lInsig / hop_length) + 2

    # Zero pad at the start and end
    insig_ = np.vstack((np.zeros((int(hop_length), nInChan)), insig_))
    insig_ = np.vstack((insig_, np.zeros((int(Nhop * hop_length - lInsig - hop_length), nInChan))))

    a = np.arange(0, Nhop - 1) * hop_length

    # Find the start and stop bins
    start_bin = int(np.round(start_freq/bin_width))
    stop_bin = int(np.round(stop_freq/bin_width)) + 1 

    # original hann window
    # window = np.hanning(self._win_len)
    # window = np.reshape(window, (np.shape(window)[0], 1))
    # window = np.vstack((window, np.zeros((self._win_len, 1))))
    # window = window * np.ones((1, nInChan))

    # adopting to DirAC window
    window = signal.windows.cosine(config["analysis"]["window_size"])
    window = np.reshape(window, (np.shape(window)[0], 1))
    window = window * np.ones((1, nInChan))

    inFramesig = np.zeros((np.shape(a)[0], np.shape(window)[0], nInChan), dtype=complex)
    inFramespec = np.zeros_like(inFramesig)

    j = 0
    for idx in (a):
        # zero pad both window and input frame to 2 * winsize for aliasing suppression
        # inFramesig[j, :, :] = np.vstack(
        #    (insig_[int(idx):int(idx) + winsize, :], np.zeros((winsize, nInChan), dtype=complex)))
        temp = insig_[int(idx):int(idx) + winsize, :]
        inFramesig[j, :temp.shape[0], :] = temp
        inFramesig[j, :, :] *= window

        #for i in range(0, nInChan):
        #    inFramespec[j, :, i] = fft(inFramesig[j, :, i])
        inFramespec[j, :, :] = fft(inFramesig[j, :, :], axis=0)
        j = j + 1
    #return inFramespec[:, :int(winsize / 2 + 1), :]  # only keeping spectrum up to nyquist
    return inFramespec[:, start_bin:stop_bin, :] # Only keep within start and stop frequencies


def get_mel_spectrogram(linear_spectra, config, start_bin, stop_bin):
    sr = config["analysis"]["sample_rate"]
    n_fft = config["analysis"]["window_size"]
    n_mels = config["analysis"]["mel"]["n_mels"]
    htk = config["dirac_analysis"]["mel"]["htk"]
    bin_width = sr / n_fft
    fmin = start_bin * bin_width
    fmax = stop_bin * bin_width  # or keep using config stop_freq

    print(f"fmin: {fmin}, fmax: {fmax}")
    print(f"mel_wts shape before slice: {librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=htk, fmin=fmin, fmax=fmax).shape}")
    print(f"linear_spectra channel 0 max: {np.abs(linear_spectra[:,:,0]).max()}")
    print(f"linear_spectra channel 1 max: {np.abs(linear_spectra[:,:,1]).max()}")

    # Build filterbank only over your frequency range, for the full n_fft
    mel_wts = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=htk, fmin=fmin, fmax=fmax).T
    # Now slice to match the spectrogram's frequency axis
    mel_wts = mel_wts[start_bin:stop_bin, :]

    assert mel_wts.shape[0] == linear_spectra.shape[1], \
        f"Filterbank {mel_wts.shape[0]} != spectrogram {linear_spectra.shape[1]}"

    mel_feat = np.zeros((linear_spectra.shape[0], n_mels, linear_spectra.shape[-1]))
    for ch_cnt in range(linear_spectra.shape[-1]):
        mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
        mel_spectra = np.dot(mag_spectra, mel_wts)
        mel_feat[:, :, ch_cnt] = mel_spectra
    return mel_feat, mel_wts


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