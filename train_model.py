import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("  STROKE PREDICTION - RANDOM FOREST")
print("=" * 50)

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.drop("id", axis=1, inplace=True)
df = df[df["gender"] != "Other"]

le = LabelEncoder()
for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    df[col] = le.fit_transform(df[col].astype(str))

imputer = SimpleImputer(strategy="median")
df["bmi"] = imputer.fit_transform(df[["bmi"]])

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print("\n🔄 Melatih model Random Forest...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_res)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi: {acc*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Stroke','Stroke'])}")

os.makedirs("model", exist_ok=True)
with open("model/stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model tersimpan!")
print("🚀 Jalankan app.py untuk memulai web server!")