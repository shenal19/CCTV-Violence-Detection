# CCTV-Violence-Detection

CCTV Violence Detection with a Temporal Confidence Gate and Automated Alerting
A final-year project that detects violent activity in CCTV-style video, classifies clips and live streams as violent or non-violent, and on a confirmed event sends an automated proof-of-concept alert (email with a snapshot) standing in for a real notification to emergency services.

The classifier is an R(2+1)D-18 spatiotemporal CNN (Kinetics-400 pretrained), fine-tuned on the RWF-2000 surveillance dataset using its official train/test split. Two robustness modules sit on top of the classifier: a temporal confidence gate (the core novelty) that turns a noisy per-window signal into stable, low-false-alarm events, and an input-quality gate that suppresses out-of-distribution input (blocked, blank, or static cameras) before the model ever sees it.

Results (RWF-2000 held-out test, multi-clip averaging)
Metric	Value
Accuracy	0.838
Precision (Violence)	0.833
Recall (Violence)	0.845
F1 (Violence)	0.839
AUC	0.927
False-positive rate	0.170
Cross-dataset generalisation (applied unchanged, no re-tuning): Hockey Fights F1 0.763 / AUC 0.770, RLVS F1 0.666 / AUC 0.752. The per-clip false-positive rate is exactly why the temporal gate exists — it confirms only sustained events, collapsing that per-clip rate into a much lower event-level alert rate. See the project report for full interpretation.

How the system works
video (clip or webcam)
  -> rolling 16-frame buffer + preprocessing (BGR->RGB, resize, Kinetics norm)
  -> input-quality gate      (too dark / too uniform / no motion -> suppressed)
  -> R(2+1)D-18              (per-window P(Violence))
  -> temporal confidence gate (EMA smooth -> N consecutive above threshold -> cooldown)
  -> on confirmed event: OpenCV overlay + async email alert (timestamp + snapshot, logged)
The architecture is modular and configuration-driven: model name, frame count, gate thresholds, and alert settings all live in configs/config.yaml, so changes are config edits rather than code changes.

Setup
1. Create the environment (Python 3.12)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
2. Install PyTorch with CUDA first
Pick the command matching your CUDA version from https://pytorch.org/get-started/locally/. This project was built and trained on an RTX 3050 Laptop (4 GB) with a CUDA-enabled build.

# example only — use the PyTorch selector to get the exact line for your machine
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
3. Install the rest
pip install -r requirements.txt
4. Verify the GPU is visible
python -m src.check_env
Expect CUDA available: True with your GPU listed. If it says False, fix the PyTorch install before continuing.

5. Set up secrets (only needed for the email alert)
cp .env.example .env      # Windows: copy .env.example .env
Then edit .env with your email credentials. Use a Gmail App Password, not your account password. .env is gitignored — never commit it.

6. Datasets (not included in the repo)
The datasets are too large to host on GitHub and are used under their research / non-commercial terms, so they are gitignored and must be downloaded separately:

RWF-2000 (primary) — keep its official train/ and val/ folders; the official split is used deliberately to avoid leakage and to allow comparison with published work.
Hockey Fights and RLVS (optional) — held-out cross-dataset generalisation tests only.
Usage
# Train (freeze-then-unfreeze, AMP, early stopping on val F1, best checkpoint saved)
python scripts/train.py

# Evaluate on the RWF-2000 test split (multi-clip averaging, confusion matrix, FPR)
python scripts/evaluate.py

# Cross-dataset generalisation test
python scripts/cross_dataset_eval.py --root HockeyFights --violence-prefix fi --nonviolence-prefix no

# Run the live inference demo on a clip or the webcam
python scripts/infer.py --source path/to/clip.mp4
python scripts/infer.py --source webcam

# Send one test alert email (verifies .env credentials)
python -m src.alerter --test
Alerts fire only when alert.enabled: true in the config and .env credentials are valid and the gate confirms a real event. With alerts disabled (the default), the alert call is a safe no-op, so normal testing never sends email.

Project layout
configs/config.yaml        all tunable settings
src/config.py              loads config + secrets
src/check_env.py           GPU / environment check
src/dataset.py             RWF dataset + dataloaders (sampling, augmentation)
src/model.py               R(2+1)D-18 builder + freeze/unfreeze helper
src/gate.py                temporal confidence gate (core novelty)
src/alerter.py             email alert module (async, logged, retrying)
scripts/train.py           training loop
scripts/evaluate.py        RWF-2000 evaluation
scripts/cross_dataset_eval.py  cross-dataset generalisation test
scripts/infer.py           live inference pipeline + input-quality gate + alert integration
scripts/test_gate.py       synthetic-stream unit tests for the gate
data/ , outputs/           datasets, checkpoints, logs, metrics (all gitignored)
Notes
Academic proof-of-concept. The email alert stands in for real emergency dispatch. A responsible real deployment would keep a human verifier in the loop rather than dispatching automatically; this is discussed in the report's ethics section.
Scope. The project deliberately delivers one accurate, robust, working detector and demo rather than a comparative model study.
