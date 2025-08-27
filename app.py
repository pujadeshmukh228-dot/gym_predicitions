from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# -------------------------------
# Load Model + Encoders + Columns
# -------------------------------
model = joblib.load("model.pkl")
encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load dataset to calculate accuracy
df = pd.read_csv("dataset.csv")
# Apply same encoding on dataset
for col, le in encoders.items():
    if col in df.columns:
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
        df[col] = le.transform(df[col].astype(str))

X = df.drop(["churn", "member_id"], axis=1, errors="ignore")
y = df["churn"]
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html", accuracy=round(acc*100, 2))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        age = int(request.form['age'])
        gender = request.form['gender']
        city = request.form['city']
        if city == "other":
            city = request.form['city_other']
        membership_type = request.form['membership_type']
        tenure_months = int(request.form['tenure_months'])
        avg_monthly_visits = int(request.form['avg_monthly_visits'])
        payment_method = request.form['payment_method']

        # Build DataFrame
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "city": city,
            "membership_type": membership_type,
            "tenure_months": tenure_months,
            "avg_monthly_visits": avg_monthly_visits,
            "payment_method": payment_method
        }])

        # Encode categorical safely
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].apply(
                    lambda x: x if x in le.classes_ else "unknown"
                )
                input_df[col] = le.transform(input_df[col].astype(str))

        # Reorder columns to match training
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        churn_prob = model.predict_proba(input_df)[0][1]
        if churn_prob > 0.7:
            prediction = "⚠️ High chance of Churn"
            prediction_class = "churn"
        else:
            prediction = "✅ Member likely to Stay"
            prediction_class = "stay"

        # Dummy Explainability
        reasons = [
            {"feature": "avg_monthly_visits", "impact": 0.45 if churn_prob > 0.7 else -0.25},
            {"feature": "tenure_months", "impact": 0.28 if churn_prob > 0.7 else -0.10},
            {"feature": "membership_type", "impact": -0.12 if churn_prob > 0.7 else 0.15}
        ]

        return render_template(
            "index.html",
            accuracy=round(acc*100, 2),
            prediction_text=f"{prediction} (Churn Probability: {churn_prob*100:.1f}%)",
            prediction_class=prediction_class,
            reasons=reasons
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", prediction_class="churn")

if __name__ == "__main__":
    app.run(debug=True)
