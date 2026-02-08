import numpy as np
def window_signal(signal, window_size, step):
    windows = []
    for i in range(0, len(signal) - window_size, step):
        windows.append(signal[i:i + window_size])
    return np.array(windows)
