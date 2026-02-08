#%%
import numpy as np
import matplotlib.pyplot as plt

def generate_current(
    f=50,
    fs=10000,
    duration=1,
    harmonics=[(3, 0.05), (5, 0.03)],
    noise=0.01,
    drift=0.0,
    rng=None
):
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(0, duration, 1/fs)
    i = np.sin(2 * np.pi * f * t)

    for h, amp in harmonics:
        i += amp * np.sin(2 * np.pi * h * f * t)

    # degradation drift (thermal, load stress)
    i *= (1 + drift * t)

    i += noise * rng.standard_normal(len(t))
    return t, i

def generate_current_signal(
    f=50,
    fs=10000,
    duration=10,
    harmonics=None,
    noise=0.01,
    drift=0.0,
    rng=None
):
    if harmonics is None:
        harmonics = []
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(0, duration, 1/fs)
    signal = np.sin(2 * np.pi * f * t)

    for h, amp in harmonics:
        signal += amp * np.sin(2 * np.pi * h * f * t)

    # gradual degradation
    signal *= (1 + drift * t)

    # noise (switching + sensor)
    signal += noise * rng.standard_normal(len(t))

    return signal

#%%

# # Normal
# t, normal = generate_current()

# # Degrading
# t, degrading = generate_current(
#     harmonics=[(3, 0.1), (5, 0.07)],
#     drift=-0.15
# )

# # Fault
# t, fault = generate_current(
#     harmonics=[(3, 0.3), (5, 0.2)],
#     drift=-0.4,
#     noise=0.05
# )

# plt.subplot(3,1,1)
# plt.plot(t,normal)
# plt.subplot(3,1,2)
# plt.plot(t,degrading)
# plt.subplot(3,1,3)
# plt.plot(t,fault)

