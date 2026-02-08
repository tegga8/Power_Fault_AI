import argparse
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from features import build_feature_matrix


def load_dataset(path):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    metadata = {key: data[key] for key in data.files if key not in {"X", "y"}}
    return X, y, metadata


def train_model(X, y, fs, test_size=0.2, seed=7):
    features = build_feature_matrix(X, fs)
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report


def save_artifacts(model, report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "fault_classifier.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    report_path = output_dir / "metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    return model_path, report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fault classifier.")
    parser.add_argument("--data", default="data/data.npz", help="Dataset npz path.")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, metadata = load_dataset(args.data)
    fs = int(metadata.get("fs", 10000))

    model, report = train_model(X, y, fs, test_size=args.test_size, seed=args.seed)
    save_artifacts(model, report, args.output_dir)


if __name__ == "__main__":
    main()
