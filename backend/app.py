"""
Flask backend for HAR prediction.
Run from Minor_HAR folder: python backend/app.py
Endpoint: POST /predict  →  { "data": [[x,y,z], ...200 samples...] }
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow browser requests from frontend

# ── Load model & helpers on startup ───────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading model...")
model = load_model(os.path.join(BASE, "har_model.keras"))
print("Model loaded.")

with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(BASE, "activity_names.pkl"), "rb") as f:
    activity_names = pickle.load(f)

# Load the same scaler used during training for consistent normalization
scaler_path = os.path.join(BASE, "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")
else:
    scaler = None
    print("WARNING: scaler.pkl not found — will fit per-window (less accurate).")

FRAME_SIZE = 200  # 20 Hz × 10 seconds

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "HAR backend running", "frame_size": FRAME_SIZE})


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
        "confidence": 0.94,
        "all_probs": { "Walking": 0.94, "Jogging": 0.03, ... }
    }
    """
    try:
        body = request.get_json(force=True)
        data = np.array(body["data"], dtype=np.float32)  # shape: (N, 3)

        if data.shape[0] < FRAME_SIZE:
            return jsonify({"error": f"Need {FRAME_SIZE} samples, got {data.shape[0]}"}), 400

        # Use last FRAME_SIZE rows in case client sends more
        data = data[-FRAME_SIZE:]

        # Normalize (use saved scaler if available, otherwise fit per-window)
        if scaler is not None:
            data = scaler.transform(data)  # shape: (200, 3)
        else:
            from sklearn.preprocessing import StandardScaler as _SS
            _scaler = _SS()
            data = _scaler.fit_transform(data)  # shape: (200, 3)

        # Reshape to (1, 200, 3, 1) for Conv2D
        X = data.reshape(1, FRAME_SIZE, 3, 1)

        # Predict
        probs = model.predict(X, verbose=0)[0]  # shape: (18,)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        activity = activity_names.get(pred_idx, f"Class {pred_idx}")

        # Build full probability dict
        all_probs = {
            activity_names.get(i, f"Class {i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        return jsonify({
            "activity": activity,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/activities", methods=["GET"])
def activities():
    return jsonify(activity_names)


if __name__ == "__main__":
    print("\n HAR Backend running at http://localhost:5000")
    print("Open frontend/index.html in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)