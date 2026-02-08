import argparse
from pathlib import Path

import numpy as np

from signals import generate_current_signal
from windowing import window_signal


def build_dataset(
    fs=10000,
    window_size=1000,
    step=500,
    duration=10,
    seed=7,
):
    rng = np.random.default_rng(seed)
    X = []
    y = []

    normal_signal = generate_current_signal(
        harmonics=[(3, 0.05), (5, 0.03)],
        noise=0.01,
        drift=0.0,
        duration=duration,
        fs=fs,
        rng=rng,
    )
    normal_windows = window_signal(normal_signal, window_size, step)
    X.extend(normal_windows)
    y.extend([0] * len(normal_windows))

    degrading_signal = generate_current_signal(
        harmonics=[(3, 0.12), (5, 0.08)],
        noise=0.02,
        drift=0.12,
        duration=duration,
        fs=fs,
        rng=rng,
    )
    degrading_windows = window_signal(degrading_signal, window_size, step)
    X.extend(degrading_windows)
    y.extend([1] * len(degrading_windows))

    fault_signal = generate_current_signal(
        harmonics=[(3, 0.35), (5, 0.25)],
        noise=0.05,
        drift=0.4,
        duration=duration,
        fs=fs,
        rng=rng,
    )
    fault_windows = window_signal(fault_signal, window_size, step)
    X.extend(fault_windows)
    y.extend([2] * len(fault_windows))

    X = np.array(X)
    y = np.array(y)
    metadata = {
        "fs": fs,
        "window_size": window_size,
        "step": step,
        "classes": np.array(["Normal", "Degrading", "Fault"]),
    }
    return X, y, metadata


def save_dataset(path, **kwargs):
    X, y, metadata = build_dataset(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y, **metadata)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate power fault dataset.")
    parser.add_argument("--output", default="data/data.npz", help="Output npz path.")
    parser.add_argument("--fs", type=int, default=10000)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    save_dataset(
        args.output,
        fs=args.fs,
        window_size=args.window_size,
        step=args.step,
        duration=args.duration,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
