"""
Flask backend for HAR prediction, Auth, MongoDB, and AI Reporting.
(Trigger reload for activity_names.pkl)
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

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from keras.models import load_model
from scipy.signal import butter, sosfilt
from bson.objectid import ObjectId
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

# Integrations
import jwt
from pymongo import MongoClient
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow all origins — we use Authorization header, not cookies

# ── Config & External Services ────────────────────────────────────────────────
FRAME_SIZE           = 60
NUM_CHANNELS         = 8
CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_WINDOW     = 5
TEMPERATURE          = 1.5   # Temperature scaling for calibrated probabilities
EMA_ALPHA            = 0.4   # Exponential moving average weight for new predictions

# MET Values for Calories (MET * weight_kg * hours)
MET_VALUES = {
    "Still": 1.3,
    "Walking": 3.5,
    "Jogging": 7.5,
    "Stairs": 8.0,
    "Hand Activity": 2.0,
    "Sports": 6.0,
    "Uncertain": 1.0
}

# ── Intensity-based activity hierarchy thresholds ─────────────────────────────
# total_var = accel_var + gyro_var  (sum of per-axis variances)
# These thresholds gate which activities are allowed at a given motion level,
# preventing slight wrist movements from being classified as Stairs / Jogging.
INTENSITY_SLIGHT   = 0.8    # <= this: only Hand Activity  (light wrist / hand motion)
INTENSITY_MODERATE = 3.0    # <= this: Walking & Stairs also allowed
INTENSITY_HIGH     = 8.0    # <= this: Jogging also allowed; above: Sports allowed

ACTIVITIES_SLIGHT   = {"Still", "Hand Activity", "Uncertain"}
ACTIVITIES_MODERATE = {"Still", "Hand Activity", "Walking", "Stairs", "Uncertain"}
ACTIVITIES_HIGH     = {"Still", "Hand Activity", "Walking", "Stairs", "Jogging", "Uncertain"}

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

# ── Custom loss function (must be defined before loading model) ────────────────
NUM_CLASSES_MODEL = 6  # Must match training config

try:
    import keras.ops as ops
except ImportError:
    import keras.backend as ops

def focal_loss_fn(y_true, y_pred):
    """Focal loss — required to load the trained model."""
    gamma, alpha = 2.0, 0.25
    y_true = ops.cast(y_true, 'int32')
    y_true_one_hot = ops.one_hot(ops.reshape(y_true, [-1]), NUM_CLASSES_MODEL)
    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -y_true_one_hot * ops.log(y_pred)
    weight = alpha * y_true_one_hot * ops.power(1.0 - y_pred, gamma)
    loss = weight * cross_entropy
    return ops.sum(loss, axis=-1)

# ── Load model & helpers on startup ───────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading model...")
model = load_model(os.path.join(BASE, "har_model.keras"),
                   custom_objects={"focal_loss_fn": focal_loss_fn})
print("  [OK] Model loaded.")

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
recent_probs = None  # EMA probability accumulator

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

def _temperature_scale(logits, temperature=TEMPERATURE):
    """Apply temperature scaling for better calibrated probabilities."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / np.sum(exp_scaled)

def _ema_update(old_probs, new_probs, alpha=EMA_ALPHA):
    """Exponential moving average on prediction probabilities."""
    if old_probs is None:
        return new_probs.copy()
    return alpha * new_probs + (1.0 - alpha) * old_probs

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
        if "access_token" in request.json:
            import requests
            access_token = request.json.get("access_token")
            user_info_url = f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={access_token}"
            resp = requests.get(user_info_url)
            if resp.status_code != 200:
                raise Exception("Invalid access token")
            user_info = resp.json()
            email = user_info["email"]
            name = user_info.get("name", "")
        else:
            print(f"Verifying token for Client ID: {GOOGLE_CLIENT_ID[:10]}...")
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
    except Exception as e:
        print(f"Google Auth Error: {str(e)}")
        return jsonify({"error": str(e), "status": "unauthorized"}), 401


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
        is_table_still = (accel_var < 0.12) and (gyro_var < 0.12)

        if scaler is not None:
            data_processed = scaler.transform(data_processed)

        global recent_probs
        probs = np.zeros(len(activity_names))
        
        if is_table_still:
            activity = "Still"
            confidence = 1.0
            still_idx = next((k for k, v in activity_names.items() if v == "Still"), 0)
            probs[still_idx] = 1.0
            recent_probs = probs.copy()
        else:
            X = data_processed.reshape(1, FRAME_SIZE, NUM_CHANNELS)
            raw_probs = model.predict(X, verbose=0)[0]
            
            # Temperature scaling for calibrated confidence
            probs = _temperature_scale(np.log(raw_probs + 1e-8))
            
            # EMA smoothing on probabilities
            recent_probs = _ema_update(recent_probs, probs)
            
            # Use EMA-smoothed probs for final decision
            pred_idx = int(np.argmax(recent_probs))
            confidence = float(recent_probs[pred_idx])
            activity = activity_names.get(pred_idx, f"Class {pred_idx}")

            if confidence < CONFIDENCE_THRESHOLD:
                activity = "Uncertain"

            # ── Intensity-based activity gating ───────────────────────────
            # Prevent high-intensity labels when the actual motion is slight.
            total_var = accel_var + gyro_var
            if total_var < INTENSITY_SLIGHT:
                allowed = ACTIVITIES_SLIGHT
            elif total_var < INTENSITY_MODERATE:
                allowed = ACTIVITIES_MODERATE
            elif total_var < INTENSITY_HIGH:
                allowed = ACTIVITIES_HIGH
            else:
                allowed = None          # everything permitted

            if allowed and activity not in allowed:
                # Pick the highest-probability activity from the allowed set
                best_idx, best_prob = None, -1.0
                for idx, name in activity_names.items():
                    if name in allowed and recent_probs[idx] > best_prob:
                        best_prob = recent_probs[idx]
                        best_idx = idx
                if best_idx is not None:
                    activity = activity_names[best_idx]
                    confidence = float(recent_probs[best_idx])

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
        
    hourly_stats = {hour: collections.Counter() for hour in range(24)}
    total_counts = collections.Counter()
    
    for log in logs:
        act = log["activity"]
        # Ensure timestamp is a datetime object
        ts = log["timestamp"]
        if isinstance(ts, datetime):
            hour = ts.hour
            hourly_stats[hour][act] += 1
            total_counts[act] += 1
            
    # Convert window counts to minutes (3 seconds per window)
    stats = {
        "totals": {act: round((count * 3) / 60, 2) for act, count in total_counts.items()},
        "hourly": {}
    }
    for hour in range(24):
        # Only include activities with > 0 minutes for efficiency
        hour_data = {act: round((count * 3) / 60, 2) for act, count in hourly_stats[hour].items() if count > 0}
        stats["hourly"][str(hour)] = hour_data
        
    total_calories = 0
    for act, mins in stats["totals"].items():
        met = MET_VALUES.get(act, 1.0)
        total_calories += met * 70 * (mins / 60.0) # Assuming 70kg average
    
    stats["total_calories"] = round(total_calories, 1)

    summary_text = f"User {current_user['name']} had the following activities today: "
    for act, mins in stats["totals"].items():
        summary_text += f"{act}: {mins:.1f} minutes, "
    summary_text += f"Total estimated calories burned: {stats['total_calories']} kcal."
        
    prompt = (
        f"You are an expert AI health and fitness assistant. Based on the following raw sensor data summary for today, "
        f"write a comprehensive, highly professional, and encouraging daily health report for the user. "
        f"Your report must include:\n"
        f"1. A clear summary of the physical activities the user has engaged in throughout the day.\n"
        f"2. Insights into their movement patterns, highlighting any positive trends or areas for improvement.\n"
        f"3. A mention of their calorie burn ({stats['total_calories']} kcal) and how it relates to their activity levels.\n"
        f"4. Constructive, actionable suggestions for better health, posture, or maintaining an active lifestyle.\n"
        f"5. Any other relevant wellness advice based on the data provided.\n"
        f"Keep the tone supportive and clinical. DO NOT use any markdown characters (like **, ###, or *). Use plain text, uppercase words for headings, and standard dashes (-) for bullets to ensure the text is clean and professional.\n"
        f"Data: {summary_text}"
    )
    
    try:
        gen_model = genai.GenerativeModel("gemini-flash-latest")
        response = gen_model.generate_content(prompt)
        
        report = {
            "user_id": current_user["_id"],
            "date": today,
            "report_text": response.text,
            "stats": stats,
            "shared_with": []
        }
        
        existing = db.reports.find_one({"user_id": current_user["_id"], "date": today})
        if existing:
            db.reports.update_one({"_id": existing["_id"]}, {"$set": {"report_text": response.text, "stats": stats}})
        else:
            db.reports.insert_one(report)
            
        return jsonify({"message": "Report generated", "report": response.text})
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500


@app.route("/api/reports", methods=["GET"])
@token_required
def get_reports(current_user):
    own_reports = list(db.reports.find({"user_id": current_user["_id"]}).sort("date", -1))
    shared_reports = list(db.reports.find({"shared_with": current_user["email"]}).sort("date", -1))
    
    for r in shared_reports:
        sender = db.users.find_one({"_id": r["user_id"]})
        if sender:
            r["sender_name"] = sender.get("name", "Unknown User")
            r["sender_email"] = sender.get("email", "")
        else:
            r["sender_name"] = "Unknown User"
            r["sender_email"] = ""
            
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

@app.route("/api/reports/<report_id>/pdf", methods=["GET"])
@token_required
def generate_pdf_report(current_user, report_id):
    try:
        report = db.reports.find_one({"_id": ObjectId(report_id)})
        if not report:
            return jsonify({"error": "Report not found"}), 404
            
        if str(report["user_id"]) != str(current_user["_id"]) and current_user["email"] not in report.get("shared_with", []):
            return jsonify({"error": "Unauthorized"}), 403
            
        owner = db.users.find_one({"_id": report["user_id"]})
        owner_name = owner["name"] if owner else "Unknown User"
        
        class PDF(FPDF):
            def header(self):
                self.set_font("helvetica", "B", 20)
                self.set_text_color(41, 128, 185)
                self.cell(0, 15, "Official Health & Activity Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_line_width(0.5)
                self.set_draw_color(41, 128, 185)
                self.line(10, 25, 200, 25)
                self.ln(10)
                
            def footer(self):
                self.set_y(-15)
                self.set_font("helvetica", "I", 8)
                self.set_text_color(128, 128, 128)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

        pdf = PDF()
        pdf.add_page()
        
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(50, 10, "Patient Name:")
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 10, owner_name, new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(40, 10, "Date of Report:")
        pdf.set_font("helvetica", "", 12)
        date_str = report["date"].strftime("%B %d, %Y") if isinstance(report["date"], datetime) else str(report["date"])
        pdf.cell(0, 10, date_str, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Summary Widgets
        calories = report.get("stats", {}).get("total_calories", 0)
        active_time = sum(report.get("stats", {}).get("totals", {}).values()) if report.get("stats") else 0
        intensity = "High" if calories > 450 else ("Moderate" if calories >= 240 else "Low")
        
        pdf.set_font("helvetica", "B", 11)
        pdf.set_fill_color(245, 247, 250)
        pdf.cell(60, 12, f"Energy: {calories} kcal", border=1, align="C", fill=True)
        pdf.cell(60, 12, f"Active Time: {int(active_time)} min", border=1, align="C", fill=True)
        pdf.cell(60, 12, f"Intensity: {intensity}", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 14)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 10, "AI Clinical Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        # Clean text for PDF (fpdf2 core fonts only support latin-1)
        report_text = report.get("report_text", "")
        clean_text = report_text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, clean_text)
        pdf.ln(10)
        
        stats = report.get("stats", {})
        if stats and "totals" in stats:
            pdf.set_font("helvetica", "B", 14)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 10, "Activity Breakdown", new_x="LMARGIN", new_y="NEXT")
            
            totals = stats["totals"]
            labels = list(totals.keys())
            values = list(totals.values())
            
            if sum(values) > 0:
                plt.figure(figsize=(6, 6))
                plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
                plt.title("Daily Activity Distribution")
                
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', bbox_inches='tight')
                img_buf.seek(0)
                plt.close()
                
                pdf.image(img_buf, x=55, w=100)
                pdf.ln(5)
                
                hourly = stats.get("hourly", {})
                if hourly:
                    hours = [str(i) for i in range(24)]
                    plt.figure(figsize=(8, 4))
                    bottoms = [0] * 24
                    
                    colors = ['#5e81ac', '#bf616a', '#d08770', '#a3be8c', '#b48ead', '#ebcb8b', '#88c0d0']
                    color_idx = 0
                    
                    for act in labels:
                        act_mins = [hourly.get(h, {}).get(act, 0) for h in hours]
                        if sum(act_mins) > 0:
                            plt.bar(hours, act_mins, bottom=bottoms, label=act, color=colors[color_idx % len(colors)])
                            bottoms = [bottoms[i] + act_mins[i] for i in range(24)]
                            color_idx += 1
                            
                    plt.title("Hourly Activity Timeline")
                    plt.xlabel("Hour of Day")
                    plt.ylabel("Minutes")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    
                    img_buf_bar = io.BytesIO()
                    plt.savefig(img_buf_bar, format='png', bbox_inches='tight')
                    img_buf_bar.seek(0)
                    plt.close()
                    
                    pdf.add_page()
                    pdf.set_font("helvetica", "B", 14)
                    pdf.set_text_color(44, 62, 80)
                    pdf.cell(0, 10, "Activity Timeline", new_x="LMARGIN", new_y="NEXT")
                    pdf.image(img_buf_bar, x=10, w=180)
                    pdf.ln(10)
                
            pdf.set_font("helvetica", "B", 12)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(0, 10, "Detailed Minutes:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 11)
            pdf.set_text_color(0, 0, 0)
            for act, mins in totals.items():
                pdf.cell(0, 7, f"- {act}: {mins} mins", new_x="LMARGIN", new_y="NEXT")
                
        pdf_bytes = pdf.output()
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"Health_Report_{owner_name.replace(' ', '_')}.pdf"
        )
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "=" * 50)
    port = int(os.getenv("PORT", 5000))
    debug_mode = os.getenv("DEBUG", "True").lower() == "true"
    print(f"  HAR Backend running at http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)