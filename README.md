# Power_Fault_AI

Synthetic power fault signal generation, feature extraction, and model training.

## Quickstart

1. Generate a dataset (stores windows and labels in `data/data.npz`):

```bash
python generate_dataset.py --output data/data.npz
```

2. Train a classifier and save artifacts:

```bash
python train_model.py --data data/data.npz --output-dir artifacts
```

The artifacts directory will contain:
- `fault_classifier.pkl` - trained classifier.
- `metrics.json` - evaluation metrics on the validation split.

## Notes

- `signals.py` contains the synthetic waveform generator.
- `features.py` extracts RMS, peak, standard deviation, and THD features.
