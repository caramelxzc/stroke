import pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("  STUNTING PREDICTION - MODEL KNN")
print("=" * 50)

df = pd.read_csv('stunting.csv')
print(f"Dataset: {df.shape[0]} baris")
print(f"Distribusi: {dict(df['Status Gizi'].value_counts())}")

encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop('Status Gizi', axis=1)
y = df['Status Gizi']
print(f"Urutan fitur: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

minority_count = y_train.value_counts().min()
k = min(5, minority_count - 1)
print(f"SMOTE k={k}")
X_train_res, y_train_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print("\n🔄 Melatih model KNN...")
model = KNeighborsClassifier(
    n_neighbors=5,
    metric='manhattan',
    weights='distance',
    algorithm='ball_tree'
)
model.fit(X_train_scaled, y_train_res)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi KNN: {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
    target_names=encoders['Status Gizi'].classes_))

os.makedirs('model', exist_ok=True)
pickle.dump(model,              open('model/stunting_model.pkl',    'wb'))
pickle.dump(scaler,             open('model/stunting_scaler.pkl',   'wb'))
pickle.dump(encoders,           open('model/stunting_encoders.pkl', 'wb'))
pickle.dump(X.columns.tolist(), open('model/stunting_columns.pkl',  'wb'))
print("✅ Model stunting tersimpan!")