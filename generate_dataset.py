import numpy as np
from signals import generate_current_signal
from windowing import window_signal

fs = 10000
window_size = 1000
step = 500

X = []
y = []

# -------- NORMAL --------
normal_signal = generate_current_signal(
    harmonics=[(3, 0.05), (5, 0.03)],
    noise=0.01,
    drift=0.0
)

normal_windows = window_signal(normal_signal, window_size, step)
X.extend(normal_windows)
y.extend([0] * len(normal_windows))

# -------- DEGRADING --------
degrading_signal = generate_current_signal(
    harmonics=[(3, 0.12), (5, 0.08)],
    noise=0.02,
    drift=0.12
)

degrading_windows = window_signal(degrading_signal, window_size, step)
X.extend(degrading_windows)
y.extend([1] * len(degrading_windows))

# -------- FAULT --------
fault_signal = generate_current_signal(
    harmonics=[(3, 0.35), (5, 0.25)],
    noise=0.05,
    drift=0.4
)

fault_windows = window_signal(fault_signal, window_size, step)
X.extend(fault_windows)
y.extend([2] * len(fault_windows))

X = np.array(X)
y = np.array(y)

# np.savez(
#     "/home/tejasbiradar/Desktop/Power Fault AI/data",
#     X=X,
#     y=y,
#     fs=fs,
#     window_size=window_size,
#     classes=["Normal", "Degrading", "Fault"]
# )
#%%
data = np.load("/home/tejasbiradar/Desktop/Power Fault AI/data/data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
X

#%%
import matplotlib.pyplot as plt

plt.plot(X[y==0][50])
plt.title("Normal Sample")
plt.show()

plt.plot(X[y==2][197])
plt.title("Fault Sample")
plt.show()
