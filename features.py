#%%
import numpy as np
from scipy.fft import fft
import signals

def extract_features(signal, fs):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    std = np.std(signal)

    fft_mag = np.abs(fft(signal))
    fft_mag = fft_mag[:len(fft_mag)//2]

    thd = np.sum(fft_mag[2:10]) / fft_mag[1]

    return [rms, peak, std, thd]

X = []
y = []
fs = 10000000

for s in signals.normal:
    print(s,end= " ")
    X.append(extract_features(s, fs))
    y.append(0)  # normal

for s in signals.degrading:
    X.append(extract_features(s, fs))
    y.append(1)  # degrading

for s in signals.fault:
    X.append(extract_features(s, fs))
    y.append(2)  # fault


print(X)