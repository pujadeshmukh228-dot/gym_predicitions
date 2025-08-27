import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")
print("Columns in dataset:", df.columns)

# -------------------------------
# 2. Encode Categorical Columns (Safe Encoder)
# -------------------------------
categorical_cols = ["gender", "city", "membership_type", "payment_method"]

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

    # Add "unknown" safely
    classes = list(le.classes_)
    if "unknown" not in classes:
        classes.append("unknown")
    le.classes_ = np.array(classes)   # ✅ FIX: list नाही, numpy array assign कर

    le_dict[col] = le

# -------------------------------
# 3. Features & Target
# -------------------------------
X = df.drop(["churn", "member_id"], axis=1, errors="ignore")
y = df["churn"]

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7. Save Model + Encoders + Columns
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("🎉 Model, encoders, and columns saved successfully!")
