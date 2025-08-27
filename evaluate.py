import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("dataset.csv")

# Load trained model
model = joblib.load("model/model.pkl")

# Drop ID-like columns if present
for col in ["member_id", "id", "memberID"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Define target
TARGET = "churn"

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Predictions
y_pred = model.predict(X)

# Accuracy
acc = accuracy_score(y, y_pred)
print(f"✅ Model Accuracy: {acc:.2f}")

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y, y_pred))



# Confusion Matrix
print("\n🌀 Confusion Matrix:")
print(confusion_matrix(y, y_pred))

