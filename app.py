"""
Stroke Prediction Web App - Flask Backend
Jalankan: python app.py
"""

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model (dibuat dari train_model.py)
MODEL_PATH = "model/stroke_model.pkl"
SCALER_PATH = "model/scaler.pkl"

model = None
scaler = None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Model berhasil dimuat!")
    else:
        print("⚠️  Model belum dilatih. Jalankan train_model.py terlebih dahulu.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ambil data dari request
        gender       = 1 if data["gender"] == "Male" else 0
        age          = float(data["age"])
        hypertension = int(data["hypertension"])
        heart        = int(data["heart_disease"])
        married      = 1 if data["ever_married"] == "Yes" else 0
        glucose      = float(data["avg_glucose_level"])
        bmi          = float(data["bmi"])

        work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
        smoke_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}
        residence_map = {"Urban": 1, "Rural": 0}

        work      = work_map.get(data["work_type"], 0)
        smoke     = smoke_map.get(data["smoking_status"], 0)
        residence = residence_map.get(data["residence_type"], 1)

        features = np.array([[gender, age, hypertension, heart, married,
                               work, residence, glucose, bmi, smoke]])

        if model is None:
            # Demo mode jika model belum ada
            risk_score = min(
                (age / 100 * 40) +
                (hypertension * 15) +
                (heart * 20) +
                (max(0, glucose - 100) / 10) +
                (max(0, bmi - 25) * 0.5) +
                (smoke * 5), 95
            )
            probability = round(risk_score, 1)
            prediction = 1 if probability >= 40 else 0
        else:
            features_scaled = scaler.transform(features)
            prediction = int(model.predict(features_scaled)[0])
            probability = round(float(model.predict_proba(features_scaled)[0][1]) * 100, 1)

        risk_level = "Rendah"
        if probability >= 65:
            risk_level = "Tinggi"
        elif probability >= 40:
            risk_level = "Sedang"

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "message": "Risiko stroke terdeteksi. Segera konsultasikan dengan dokter." if prediction == 1
                       else "Risiko stroke rendah. Pertahankan gaya hidup sehat!",
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "version": "1.0.0"
    })

if __name__ == "__main__":
    load_model()
    print("🚀 Server berjalan di http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
