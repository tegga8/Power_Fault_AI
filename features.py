import numpy as np
from scipy.fft import fft


def extract_features(signal, fs):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    std = np.std(signal)

    fft_mag = np.abs(fft(signal))
    fft_mag = fft_mag[: len(fft_mag) // 2]
    harmonic_band = fft_mag[2:10]
    fundamental = fft_mag[1] + 1e-12
    thd = np.sum(harmonic_band) / fundamental

    return np.array([rms, peak, std, thd])


def build_feature_matrix(windows, fs):
    return np.array([extract_features(window, fs) for window in windows])
