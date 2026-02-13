from flask import Flask, render_template, request
import pickle
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Load trained model
model = pickle.load(open("random_forest_model.pkl", "rb"))

# Feature order MUST match training dataset
feature_order = [
    'age_first_funding_year',
    'age_last_funding_year',
    'age_first_milestone_year',
    'age_last_milestone_year',
    'relationships',
    'funding_rounds',
    'funding_total_usd',
    'milestones',
    'is_CA','is_NY','is_MA','is_TX','is_otherstate',
    'is_software','is_web','is_mobile','is_enterprise',
    'is_advertising','is_gamesvideo','is_ecommerce',
    'is_biotech','is_consulting','is_othercategory',
    'has_VC','has_angel','has_roundA','has_roundB',
    'has_roundC','has_roundD',
    'avg_participants','is_top500'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- LOCATION HANDLING --------
        location = request.form.get("location")

        is_CA = 1 if location == "CA" else 0
        is_NY = 1 if location == "NY" else 0
        is_MA = 1 if location == "MA" else 0
        is_TX = 1 if location == "TX" else 0
        is_otherstate = 1 if location == "OTHER" else 0

        # -------- COLLECT NUMERIC INPUTS --------
        input_data = []

        for feature in feature_order:
            if feature in ['is_CA','is_NY','is_MA','is_TX','is_otherstate']:
                continue  # handled separately

            value = request.form.get(feature)

            if value is None or value == "":
                value = 0

            input_data.append(float(value))

        # Insert location values in correct order
        input_data.insert(8, is_CA)
        input_data.insert(9, is_NY)
        input_data.insert(10, is_MA)
        input_data.insert(11, is_TX)
        input_data.insert(12, is_otherstate)

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        probability_percent = round(probability * 100, 2)

        if prediction == 1:
            result = "🚀 Startup is Likely to SUCCEED"
            status_class = "success"
        else:
            result = "❌ Startup is Likely to FAIL"
            status_class = "danger"

        return render_template(
            "result.html",
            prediction_text=result,
            probability=probability_percent,
            status_class=status_class
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        return render_template(
            "result.html",
            prediction_text="Error in Prediction",
            probability=0,
            status_class="warning"
        )

if __name__ == "__main__":
    app.run(debug=True)
