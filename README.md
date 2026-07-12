# IMU Table-Tennis Coaching System

A real-time table-tennis coaching system driven by wearable **IMU (inertial) sensors**. It
captures a player's motion, visualizes it as a live 3D skeleton in **Unity**, classifies the
stroke type with a trained neural network, and generates personalized coaching feedback with
a locally-running **large language model (LLM)** — giving players immediate, data-driven
feedback without a human coach present.

This repository is my **MSc thesis** submission work.

## Demo

▶️ **[Watch the demo video](demo/classification.mp4)** (`demo/classification.mp4`) — live
IMU stream driving the Unity skeleton with real-time stroke classification.

## What it does

The system is a four-stage real-time pipeline:

1. **Capture** — body-worn IMU sensors stream 9-DoF motion (accelerometer + gyroscope +
   orientation) over WiFi/UDP.
2. **Visualize** — a Unity avatar (Mixamo character) is driven frame-by-frame from the
   sensor stream, so the player sees their own skeleton move in real time.
3. **Classify** — a sliding-window feature extractor computes statistical features from the
   wrist IMU, and an MLP classifier predicts the stroke class (e.g. *no-stroke / forehand
   topspin / backhand drive / forehand smash*). Training uses **leave-one-subject-out
   (LOSO)** cross-validation.
4. **Coach** — a session summarizer aggregates the predictions (stroke distribution, tempo,
   confidence, weak strokes) and a local LLM turns them into a structured coaching report
   (assessment + prioritized recommendations + next-session focus).

## What it uses

- **Python** (PyTorch) for the ML pipeline (feature extraction, MLP, LOSO training,
  inference).
- **Unity** (C#) for the real-time 3D skeleton visualization, driven over **UDP**.
- **A local LLM** via an OpenAI-compatible server (e.g. LM Studio) for coaching feedback.
- **SiriusCeption** IMU hardware / controller for sensor capture and calibration.
- **TTSWING** table-tennis IMU dataset (public) for training the stroke classifier.

## Repository structure

```
pipeline/          ML pipeline
  model.py               MLP stroke classifier
  train.py               leave-one-subject-out training loop
  inference.py           real-time stroke predictor
  feature_extractor.py   sliding-window statistical features
  dataset.py, preprocessing.py, evaluation.py
  coaching.py            LLM coaching feedback
  summarizer.py          session summary builder
  signal_filter.py, kinematics.py
  realtimeapp_*.py, live_app*.py   real-time GUI applications
realtime_app/      IMU -> Unity streamer, Unity C# receiver (NewIMUFull.cs),
                   bone / T-pose config, skeleton viewer
unity/             Unity project (Assets / ProjectSettings / Packages)
checkpoints/       trained LOSO model weights (.pt) + scalers (.pkl)
notebooks/         comparison / analysis notebook
config.yaml        model, IMU, and LLM configuration
demo/              demo video
```

## Setup

```bash
pip install -r pipeline_requirements.txt      # Python dependencies
```
Open the Unity project in `unity/` with the Unity Editor to build/run the visualization.
For coaching, run a local OpenAI-compatible LLM server (e.g. LM Studio) and set its
endpoint in `config.yaml`.

## How to run

```bash
# 1) Train the stroke classifier (leave-one-subject-out)
python pipeline/train.py

# 2) Run the real-time coaching app (needs a live sensor stream + the SiriusCeption controller)
python pipeline/realtimeapp_new.py
```
Then press **Play** in the Unity project to see the live skeleton, and use the app's
"Generate Report" action to get LLM coaching feedback.

## Data (not included)

Datasets are large and are **not** distributed in this repository. The stroke classifier is
trained on the public **TTSWING** dataset (available from its original authors). Point
`config.yaml` at your own local copy of the data.

## SiriusCeption controller (`.pyd` / `.so`) — not included

The system relies on `SiriusCeption_unity_controller` (a compiled binary that handles sensor
calibration and Unity streaming). It is **proprietary and not redistributed here**. To obtain
it, **please contact the author.**

## License

MIT — see [LICENSE](LICENSE).
