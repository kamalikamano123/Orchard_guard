from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import os
from model import predict_pil   # <-- add this
from PIL import Image


app = Flask(__name__)
CORS(app)

# 📌 Connect MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["orchard_guard"]

# Public folder for frontend
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "../public")


# -------------------------
# AUTH (Signup + Login)
# -------------------------
@app.route("/api/signup", methods=["POST"])
def signup():
    users_col = db.users
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"message": "Email and password required"}), 400

    if users_col.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    users_col.insert_one({"email": email, "password": password})
    return jsonify({"message": "Signup successful"})


@app.route("/api/login", methods=["POST"])
def login():
    users_col = db.users
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users_col.find_one({"email": email, "password": password})
    if user:
        return jsonify({"message": "Login successful"})
    return jsonify({"message": "Invalid credentials"}), 401


# -------------------------
# PROFILE (per user)
# -------------------------
@app.route("/api/profile/<email>", methods=["GET", "POST"])
def profile(email):
    profiles_col = db.profiles

    if request.method == "GET":
        profile = profiles_col.find_one({"email": email}, {"_id": 0})
        if profile:
            return jsonify(profile)
        return jsonify({"message": "Profile not found"}), 404

    if request.method == "POST":
        data = request.json
        data["email"] = email
        profiles_col.update_one({"email": email}, {"$set": data}, upsert=True)
        return jsonify({"message": "Profile saved successfully"})


# -------------------------
# SCHEDULE (per user)
# -------------------------
@app.route("/api/schedule/<email>", methods=["GET", "POST"])
def schedule(email):
    schedule_col = db.schedule

    if request.method == "GET":
        schedules = list(schedule_col.find({"email": email}, {"_id": 0}))
        return jsonify(schedules)

    if request.method == "POST":
        data = request.json
        data["email"] = email
        schedule_col.insert_one(data)
        return jsonify({"message": "Schedule saved successfully"})


# -------------------------
# SETTINGS (per user)
# -------------------------
@app.route("/api/settings/<email>", methods=["GET", "POST"])
def settings(email):
    settings_col = db.settings

    if request.method == "GET":
        settings = settings_col.find_one({"email": email}, {"_id": 0})
        if settings:
            return jsonify(settings)
        return jsonify({"message": "Settings not found"}), 404

    if request.method == "POST":
        data = request.json
        data["email"] = email
        settings_col.update_one({"email": email}, {"$set": data}, upsert=True)
        return jsonify({"message": "Settings saved successfully"})

# -------------------------
# ML MODEL PREDICTION
# -------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file.stream)
        result = predict_pil(img)   # call model.py
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# -------------------------
# FRONTEND ROUTES
# -------------------------
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    return send_from_directory(PUBLIC_DIR, path)

# -------------------------
# ML MODEL PREDICTION




if __name__ == "__main__":
    app.run(debug=True)
