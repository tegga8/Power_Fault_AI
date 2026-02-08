 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 616f2354e08f888734e661403d58ea75dd93b966..64645e2c5b41e6c571ed176ae4761fe4ac5740db 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,26 @@
-# Power_Fault_AI
\ No newline at end of file
+# Power_Fault_AI
+
+Synthetic power fault signal generation, feature extraction, and model training.
+
+## Quickstart
+
+1. Generate a dataset (stores windows and labels in `data/data.npz`):
+
+```bash
+python generate_dataset.py --output data/data.npz
+```
+
+2. Train a classifier and save artifacts:
+
+```bash
+python train_model.py --data data/data.npz --output-dir artifacts
+```
+
+The artifacts directory will contain:
+- `fault_classifier.pkl` - trained classifier.
+- `metrics.json` - evaluation metrics on the validation split.
+
+## Notes
+
+- `signals.py` contains the synthetic waveform generator.
+- `features.py` extracts RMS, peak, standard deviation, and THD features.
 
EOF
)
