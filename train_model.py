import pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

print("=" * 50)
print("  STROKE PREDICTION - MODEL KNN")
print("=" * 50)

df = pd.read_csv('stroke_clean.csv')
print(f"Dataset: {df.shape[0]} baris")

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_res, y_train_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print("\n🔄 Melatih model KNN...")
model = KNeighborsClassifier(
    n_neighbors=3,
    metric='manhattan',
    weights='distance',
    algorithm='ball_tree'
)
model.fit(X_train_scaled, y_train_res)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi KNN: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['No Stroke','Stroke']))

os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/stroke_model.pkl','wb'))
pickle.dump(scaler, open('model/scaler.pkl','wb'))
print("✅ Model KNN tersimpan!")