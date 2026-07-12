# IMU Table-Tennis Coaching System

Real-time table-tennis coaching from wearable IMU sensors: capture body motion, visualize
it as a 3D skeleton in Unity, classify stroke types with a trained neural network, and
generate coaching feedback with a locally-running LLM. This repository is my MSc thesis
submission work.

## What it does
1. **Capture** — body-worn IMU sensors stream motion data.
2. **Visualize** — a Unity avatar is driven in real time from the sensor stream (UDP).
3. **Classify** — a sliding-window feature extractor + MLP classifies each stroke
   (e.g. no-stroke / forehand topspin / backhand drive / forehand smash).
4. **Coach** — a local LLM turns the session statistics into structured coaching feedback.

## Repository layout
```
pipeline/        ML pipeline: feature extraction, model, training (LOSO), inference,
                 coaching + summarizer, signal filtering, kinematics, real-time apps
realtime_app/    IMU -> Unity streamer, Unity C# receiver, bone/T-pose config, viewer
unity/           Unity project (Assets / ProjectSettings / Packages)
checkpoints/     Trained LOSO model weights (.pt) + scalers (.pkl)
notebooks/       Comparison / analysis notebooks
config.yaml      Model, IMU, and LLM configuration
```

## Not included (and why)
- **Datasets** (e.g. TTSWING and raw recordings) are large and are not distributed here.
  Point `config.yaml` at your own local copy. TTSWING is publicly available from its
  original authors.
- **`SiriusCeption_unity_controller` (`.pyd`/`.so`)** — a proprietary compiled binary
  provided by a lab collaborator for sensor calibration and Unity streaming. It is **not**
  redistributed here. Obtain it from the SiriusCeption maintainers; place it under
  `unity/Assets/Plugins/` and alongside the Python controller.

## Quick start
```bash
pip install -r pipeline/requirements.txt      # or realtime_app/requirements.txt
# train (leave-one-subject-out)
python pipeline/train.py
# real-time app (needs the SiriusCeption controller + a live sensor stream)
python pipeline/realtimeapp_new.py
```
Open the Unity project in `unity/` and press Play to see the live skeleton.

## Notes
- Classification uses statistical features from a sliding window over the wrist IMU.
- Training is leave-one-subject-out; one checkpoint + scaler per held-out subject.
- The LLM coaching module targets a local OpenAI-compatible server (see `config.yaml`).

## License
MIT — see [LICENSE](LICENSE).
