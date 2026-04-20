"""
Flask backend for HAR prediction.
Run from Minor_HAR folder: python backend/app.py
Endpoint: POST /predict  →  { "data": [[x,y,z], ...200 samples...] }
"""

import os
import csv
import pickle
import collections
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow browser requests from frontend

# ── Config ────────────────────────────────────────────────────────────────────
FRAME_SIZE           = 200    # 20 Hz × 10 seconds
CONFIDENCE_THRESHOLD = 0.6    # below this → "Uncertain"
SMOOTHING_WINDOW     = 3      # majority vote over last N predictions

# ── Load model & helpers on startup ───────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading model...")
model = load_model(os.path.join(BASE, "har_model.keras"))
print("  ✔ Model loaded.")

with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
print("  ✔ Label encoder loaded.")

with open(os.path.join(BASE, "activity_names.pkl"), "rb") as f:
    activity_names = pickle.load(f)
print("  ✔ Activity names loaded.")

# Load the same scaler used during training for consistent normalization
scaler_path = os.path.join(BASE, "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("  ✔ Scaler loaded.")
else:
    scaler = None
    print("  ⚠ scaler.pkl not found — will fit per-window (less accurate).")

# ── Prediction smoothing buffer ───────────────────────────────────────────────
recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_PATH = os.path.join(BASE, "predictions_log.csv")
# Create the log file with a header if it doesn't exist
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "predicted_activity", "confidence"])
    print(f"  ✔ Log file created → {LOG_PATH}")
else:
    print(f"  ✔ Log file exists  → {LOG_PATH}")


def _log_prediction(activity, confidence):
    """Append a single prediction row to the CSV log."""
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            activity,
            round(confidence, 4),
        ])


def _majority_vote(predictions):
    """Return the most common prediction from the deque."""
    if not predictions:
        return None
    counter = collections.Counter(predictions)
    return counter.most_common(1)[0][0]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "HAR backend running",
        "frame_size": FRAME_SIZE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
        "data": [[x1,y1,z1], [x2,y2,z2], ... ]   ← exactly 200 rows
    }
    Returns:
    {
        "activity": "Walking",
        "confidence": 0.87,
        "all_probs": { "Walking": 0.87, ... },
        "all_probabilities": [0.87, 0.03, ...],
        "smoothed_activity": "Walking",
        "status": "ok"
    }
    """
    try:
        body = request.get_json(force=True)

        if "data" not in body:
            return jsonify({
                "error": "Missing 'data' field in request body",
                "status": "error",
            }), 400

        data = np.array(body["data"], dtype=np.float32)  # shape: (N, 3)

        # ── Strict length check ───────────────────────────────────────────
        if data.shape[0] != FRAME_SIZE:
            return jsonify({
                "error": f"Expected exactly {FRAME_SIZE} samples, got {data.shape[0]}",
                "status": "error",
            }), 400

        if data.ndim != 2 or data.shape[1] != 3:
            return jsonify({
                "error": f"Each sample must have 3 axes (X, Y, Z), got shape {data.shape}",
                "status": "error",
            }), 400

        # ── Normalize ─────────────────────────────────────────────────────
        if scaler is not None:
            data = scaler.transform(data)
        else:
            from sklearn.preprocessing import StandardScaler as _SS
            _scaler = _SS()
            data = _scaler.fit_transform(data)

        # ── Predict ───────────────────────────────────────────────────────
        X = data.reshape(1, FRAME_SIZE, 3, 1)
        probs = model.predict(X, verbose=0)[0]        # shape: (num_classes,)
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        activity   = activity_names.get(pred_idx, f"Class {pred_idx}")

        # ── Confidence threshold ──────────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            activity = "Uncertain"

        # ── Smoothing (majority vote) ─────────────────────────────────────
        recent_predictions.append(activity)
        smoothed_activity = _majority_vote(recent_predictions)

        # ── Build probability dictionaries ────────────────────────────────
        all_probs = {
            activity_names.get(i, f"Class {i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }
        all_probabilities = [round(float(p), 4) for p in probs]

        # ── Log to CSV ────────────────────────────────────────────────────
        _log_prediction(smoothed_activity, confidence)

        # ── Response (backward-compatible + enhanced) ─────────────────────
        return jsonify({
            "activity":          smoothed_activity,
            "raw_activity":      activity,
            "confidence":        round(confidence, 4),
            "all_probs":         all_probs,          # dict  — existing frontend uses this
            "all_probabilities": all_probabilities,  # list  — new consumers can use this
            "smoothed_activity": smoothed_activity,
            "status":            "ok",
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/activities", methods=["GET"])
def activities():
    return jsonify(activity_names)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  HAR Backend running at http://localhost:5000")
    print("  Open frontend/index.html in your browser")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)