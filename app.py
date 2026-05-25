from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── PATH ──────────────────────────────────────────
MODEL_PATH             = "model/stroke_model.pkl"
SCALER_PATH            = "model/scaler.pkl"
ENCODERS_PATH          = "model/encoders.pkl"
COLUMNS_PATH           = "model/feature_columns.pkl"

STUNTING_MODEL_PATH    = "model/stunting_model.pkl"
STUNTING_SCALER_PATH   = "model/stunting_scaler.pkl"
STUNTING_ENCODERS_PATH = "model/stunting_encoders.pkl"
STUNTING_COLUMNS_PATH  = "model/stunting_columns.pkl"

# ── GLOBALS ───────────────────────────────────────
model           = None
scaler          = None
encoders        = None
feature_columns = None

stunting_model    = None
stunting_scaler   = None
stunting_encoders = None
stunting_columns  = None

# ── LOAD STROKE ───────────────────────────────────
def load_model():
    global model, scaler, encoders, feature_columns
    for path in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, COLUMNS_PATH]:
        if not os.path.exists(path):
            print(f"⚠️  File tidak ditemukan: {path}")
            return
    with open(MODEL_PATH,    "rb") as f: model           = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler          = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f: encoders        = pickle.load(f)
    with open(COLUMNS_PATH,  "rb") as f: feature_columns = pickle.load(f)
    print("✅ Model stroke berhasil dimuat!")

# ── LOAD STUNTING ─────────────────────────────────
def load_stunting_model():
    global stunting_model, stunting_scaler, stunting_encoders, stunting_columns
    for path in [STUNTING_MODEL_PATH, STUNTING_SCALER_PATH,
                 STUNTING_ENCODERS_PATH, STUNTING_COLUMNS_PATH]:
        if not os.path.exists(path):
            print(f"⚠️  File tidak ditemukan: {path}")
            return
    with open(STUNTING_MODEL_PATH,    "rb") as f: stunting_model    = pickle.load(f)
    with open(STUNTING_SCALER_PATH,   "rb") as f: stunting_scaler   = pickle.load(f)
    with open(STUNTING_ENCODERS_PATH, "rb") as f: stunting_encoders = pickle.load(f)
    with open(STUNTING_COLUMNS_PATH,  "rb") as f: stunting_columns  = pickle.load(f)
    print("✅ Model stunting berhasil dimuat!")

# ── ROUTES ────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "stroke_model":   model is not None,
        "stunting_model": stunting_model is not None,
        "version": "1.0.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Request body kosong."}), 400

        if model is None:
            return jsonify({"status": "error",
                "message": "Model stroke belum dimuat."}), 503

        def encode(col, val):
            return int(encoders[col].transform([str(val)])[0])

        row = {
            "gender":            encode("gender",         data["gender"]),
            "age":               float(data["age"]),
            "hypertension":      int(data["hypertension"]),
            "heart_disease":     int(data["heart_disease"]),
            "ever_married":      encode("ever_married",   data["ever_married"]),
            "work_type":         encode("work_type",      data["work_type"]),
            "Residence_type":    encode("Residence_type", data["residence_type"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi":               float(data["bmi"]),
            "smoking_status":    encode("smoking_status", data["smoking_status"]),
        }
        features        = np.array([[row[col] for col in feature_columns]])
        features_scaled = scaler.transform(features)
        prediction      = int(model.predict(features_scaled)[0])

        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(features_scaled)[0][1]) * 100, 1)
        else:
            probability = 80.0 if prediction == 1 else 20.0

        risk_level = "Rendah"
        if probability >= 65:
            risk_level = "Tinggi"
        elif probability >= 40:
            risk_level = "Sedang"

        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "message": "Risiko stroke terdeteksi. Segera konsultasikan dengan dokter."
                       if prediction == 1 else "Risiko stroke rendah. Pertahankan gaya hidup sehat!",
            "status": "success"
        })

    except Exception as e:
        print(f"[ERROR stroke] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict-stunting", methods=["POST"])
def predict_stunting():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Request body kosong."}), 400

        if stunting_model is None:
            return jsonify({"status": "error",
                "message": "Model stunting belum dimuat."}), 503

        umur   = float(data.get("umur", 0))
        tinggi = float(data.get("tinggi_badan", 0))
        gender = data.get("jenis_kelamin", "")

        if not (0 <= umur <= 36):
            raise ValueError("Umur harus antara 0–36 bulan.")
        if not (45 <= tinggi <= 110):
            raise ValueError("Tinggi badan harus antara 45–110 cm.")
        if gender not in stunting_encoders['Jenis Kelamin'].classes_:
            raise ValueError("Jenis kelamin tidak valid.")

        gender_enc = int(stunting_encoders['Jenis Kelamin'].transform([gender])[0])
        row = {
            "Umur":          umur,
            "Jenis Kelamin": gender_enc,
            "Tinggi Badan":  tinggi,
        }
        features        = np.array([[row[col] for col in stunting_columns]])
        features_scaled = stunting_scaler.transform(features)
        pred_idx        = int(stunting_model.predict(features_scaled)[0])
        pred_label      = stunting_encoders['Status Gizi'].inverse_transform([pred_idx])[0]

        if hasattr(stunting_model, "predict_proba"):
            proba      = stunting_model.predict_proba(features_scaled)[0]
            confidence = round(float(max(proba)) * 100, 1)
        else:
            confidence = None

        pesan = {
            "normal":           "Tinggi badan anak sesuai usia. Pertahankan pola makan bergizi!",
            "stunted":          "Anak mengalami stunting. Konsultasikan dengan dokter/ahli gizi.",
            "severely stunted": "Anak mengalami stunting parah. Segera konsultasi dengan tenaga medis.",
            "tinggi":           "Tinggi badan anak di atas rata-rata. Tetap jaga pola makan sehat!"
        }
        warna = {
            "normal": "green", "stunted": "amber",
            "severely stunted": "red", "tinggi": "blue"
        }

        return jsonify({
            "status":     "success",
            "prediction": pred_label,
            "confidence": confidence,
            "color":      warna.get(pred_label, "gray"),
            "message":    pesan.get(pred_label, "Konsultasikan dengan dokter.")
        })

    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 422
    except Exception as e:
        print(f"[ERROR stunting] {e}")
        return jsonify({"status": "error", "message": "Terjadi kesalahan pada server."}), 500

if __name__ == "__main__":
    load_model()
    load_stunting_model()
    print("🚀 Server berjalan di http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)