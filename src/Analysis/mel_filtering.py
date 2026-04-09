# This utility script uses mel spectrograms that can be applied
# at feature extraction to compress the size of the feature tensor
# by taking into account human auditory perception.

import numpy as np
from librosa.filters import mel

class MelFeatureMapper:
    def __init__(self, sr, bin_size, n_mels=64, log_mel=True):
        
        # These parameters can be initialised from the 
        # import .yaml config file.
        self.sr = sr
        self.bin_size = bin_size
        self.n_mels = n_mels
        self.log_mel = log_mel
        
        # Convert the operand spectrogram bin size to a n_fft value:
        self.n_fft = (bin_size - 1) * 2

        # Create mel filterbank (shape: [n_mels, n_freq_bins])
        self.mel_filterbank = mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels)

    def map(self, spectrogram):

        # Apply the mel filterbank to the chosen spectrogram
        # such ITD or ILD cues.
        mel_spec = self.mel_filterbank @ spectrogram

        return mel_spec
