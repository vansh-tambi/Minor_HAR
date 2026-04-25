"""
Flask backend for HAR prediction, Auth, MongoDB, and AI Reporting.
Run from Minor_HAR folder: python backend/app.py
"""

import os
import csv
import pickle
import collections
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from scipy.signal import butter, sosfilt
from bson.objectid import ObjectId

# Integrations
import jwt
from pymongo import MongoClient
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Config & External Services ────────────────────────────────────────────────
FRAME_SIZE           = 60
NUM_CHANNELS         = 8
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_WINDOW     = 3

# MongoDB
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.har_database

# Auth & JWT
GOOGLE_CLIENT_ID = os.getenv("VITE_GOOGLE_CLIENT_ID")
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-har-key")

# Gemini
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ── Load model & helpers on startup ───────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading model...")
model = load_model(os.path.join(BASE, "har_model.keras"))
print("  ✔ Model loaded.")

with open(os.path.join(BASE, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

with open(os.path.join(BASE, "activity_names.pkl"), "rb") as f:
    activity_names = pickle.load(f)

scaler_path = os.path.join(BASE, "scaler.pkl")
if os.path.exists(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = None

recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)

# ── Filter setup ──────────────────────────────────────────────────────────────
FS = 20.0
NYQ = 0.5 * FS
sos_noise = butter(3, 5.0 / NYQ, btype='low', output='sos')
sos_gravity = butter(3, 0.3 / NYQ, btype='low', output='sos')

filter_states_noise = [np.zeros((sos_noise.shape[0], 2)) for _ in range(6)]
filter_states_gravity = [np.zeros((sos_gravity.shape[0], 2)) for _ in range(3)]

def real_time_preprocess(new_samples_6d):
    filtered = np.zeros_like(new_samples_6d)
    for i in range(6):
        filtered[:, i], filter_states_noise[i] = sosfilt(sos_noise, new_samples_6d[:, i], zi=filter_states_noise[i])
    accel = filtered[:, 0:3]
    gyro = filtered[:, 3:6]
    gravity = np.zeros_like(accel)
    for i in range(3):
        gravity[:, i], filter_states_gravity[i] = sosfilt(sos_gravity, accel[:, i], zi=filter_states_gravity[i])
    body_accel = accel - gravity
    accel_mag = np.linalg.norm(body_accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    return np.hstack((body_accel, gyro, accel_mag, gyro_mag))

def _majority_vote(predictions):
    if not predictions:
        return None
    counter = collections.Counter(predictions)
    return counter.most_common(1)[0][0]

# ── Auth Decorator ────────────────────────────────────────────────────────────
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = token.split(" ")[1] # Bearer token
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            current_user = db.users.find_one({"email": data["email"]})
            if not current_user:
                raise Exception("User not found")
        except Exception as e:
            return jsonify({"error": "Token is invalid"}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "HAR backend running"})

@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    token = request.json.get("token")
    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Server missing Google Client ID"}), 500
        
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = idinfo["email"]
        name = idinfo.get("name", "")
        
        user = db.users.find_one({"email": email})
        if not user:
            user = {"email": email, "name": name, "created_at": datetime.utcnow()}
            db.users.insert_one(user)
            
        jwt_token = jwt.encode(
            {"email": email, "exp": datetime.utcnow() + timedelta(days=7)}, 
            JWT_SECRET, 
            algorithm="HS256"
        )
        return jsonify({"token": jwt_token, "user": {"email": email, "name": name}})
    except ValueError:
        return jsonify({"error": "Invalid token"}), 401


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)
        if "data" not in body:
            return jsonify({"error": "Missing 'data' field", "status": "error"}), 400

        data = np.array(body["data"], dtype=np.float32)
        if data.shape[0] != FRAME_SIZE or data.ndim != 2 or data.shape[1] != 6:
            return jsonify({"error": "Invalid shape", "status": "error"}), 400

        data_processed = real_time_preprocess(data)
        
        accel_var = np.sum(np.var(data[:, 0:3], axis=0))
        gyro_var = np.sum(np.var(data[:, 3:6], axis=0))
        is_table_still = (accel_var < 0.15) and (gyro_var < 0.15)

        if scaler is not None:
            data_processed = scaler.transform(data_processed)

        probs = np.zeros(len(activity_names))
        
        if is_table_still:
            activity = "Still"
            confidence = 1.0
            still_idx = next((k for k, v in activity_names.items() if v == "Still"), 0)
            probs[still_idx] = 1.0
        else:
            X = data_processed.reshape(1, FRAME_SIZE, NUM_CHANNELS)
            probs = model.predict(X, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            activity = activity_names.get(pred_idx, f"Class {pred_idx}")

            if confidence < CONFIDENCE_THRESHOLD:
                activity = "Uncertain"

        recent_predictions.append(activity)
        smoothed_activity = _majority_vote(recent_predictions)
        
        # ── DB Logging (Optional if user is authenticated) ─────────────────
        token_header = request.headers.get("Authorization")
        if token_header:
            try:
                token = token_header.split(" ")[1]
                data_jwt = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                user = db.users.find_one({"email": data_jwt["email"]})
                if user and smoothed_activity != "Uncertain":
                    db.activity_logs.insert_one({
                        "user_id": user["_id"],
                        "timestamp": datetime.utcnow(),
                        "activity": smoothed_activity,
                        "confidence": float(confidence)
                    })
            except Exception as e:
                pass # Fail silently for logging

        all_probs = {activity_names.get(i, f"Class {i}"): round(float(p), 4) for i, p in enumerate(probs)}
        return jsonify({
            "activity": smoothed_activity,
            "raw_activity": activity,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
            "smoothed_activity": smoothed_activity,
            "status": "ok",
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/reports/generate", methods=["POST"])
@token_required
def generate_report(current_user):
    if not os.getenv("GEMINI_API_KEY"):
        return jsonify({"error": "Gemini API key not configured"}), 500
        
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    logs = list(db.activity_logs.find({"user_id": current_user["_id"], "timestamp": {"$gte": today}}))
    
    if not logs:
        return jsonify({"error": "No activity logged today"}), 400
        
    activity_counts = collections.Counter([log["activity"] for log in logs])
    summary_text = f"User {current_user['name']} had the following activities today: "
    for act, count in activity_counts.items():
        minutes = (count * 3) / 60 # 3 seconds per window
        summary_text += f"{act}: {minutes:.1f} minutes, "
        
    prompt = f"You are an AI health assistant. Based on this raw sensor data summary, write a highly professional, encouraging, 2-paragraph daily health report for the user. Do not include raw numbers if they are very small, just summarize the movement patterns. Data: {summary_text}"
    
    gen_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gen_model.generate_content(prompt)
    
    report = {
        "user_id": current_user["_id"],
        "date": today,
        "report_text": response.text,
        "shared_with": []
    }
    
    existing = db.reports.find_one({"user_id": current_user["_id"], "date": today})
    if existing:
        db.reports.update_one({"_id": existing["_id"]}, {"$set": {"report_text": response.text}})
    else:
        db.reports.insert_one(report)
        
    return jsonify({"message": "Report generated", "report": response.text})


@app.route("/api/reports", methods=["GET"])
@token_required
def get_reports(current_user):
    own_reports = list(db.reports.find({"user_id": current_user["_id"]}).sort("date", -1))
    shared_reports = list(db.reports.find({"shared_with": current_user["email"]}).sort("date", -1))
    
    for r in own_reports + shared_reports:
        r["_id"] = str(r["_id"])
        r["user_id"] = str(r["user_id"])
        r["date"] = r["date"].strftime("%Y-%m-%d")
        
    return jsonify({"own": own_reports, "shared": shared_reports})


@app.route("/api/reports/share", methods=["POST"])
@token_required
def share_report(current_user):
    report_id = request.json.get("report_id")
    target_email = request.json.get("email")
    
    target_user = db.users.find_one({"email": target_email})
    if not target_user:
        return jsonify({"error": "User not found"}), 404
        
    result = db.reports.update_one(
        {"_id": ObjectId(report_id), "user_id": current_user["_id"]},
        {"$addToSet": {"shared_with": target_email}}
    )
    
    if result.modified_count == 0:
        return jsonify({"error": "Report not found or already shared"}), 400
        
    return jsonify({"message": f"Report shared with {target_email}"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "True").lower() == "true"
    print(f"  HAR Backend running at http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)