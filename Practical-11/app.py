
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained models and metadata
dt_model = joblib.load("decision_tree_classifier.pkl")
rf_model = joblib.load("random_forest_classifier.pkl")
metadata = joblib.load("iris_metadata.pkl")

features = metadata["features"]
target_names = metadata["target_names"]
defaults = metadata["defaults"]


@app.route("/")
def home():
    return render_template(
        "index.html",
        features=features,
        defaults=defaults
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the four input measurements
        values = [float(request.form[f]) for f in features]

        # Create input DataFrame in the correct feature order
        input_data = pd.DataFrame([values], columns=features)

        # Make predictions with both models
        dt_prediction = int(dt_model.predict(input_data)[0])
        rf_prediction = int(rf_model.predict(input_data)[0])

        dt_result = target_names[dt_prediction]
        rf_result = target_names[rf_prediction]

        return render_template(
            "index.html",
            features=features,
            defaults=defaults,
            form_values=request.form,
            dt_result=dt_result,
            rf_result=rf_result
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            features=features,
            defaults=defaults,
            error="Please enter valid numeric values in all four fields."
        )


if __name__ == "__main__":
    app.run(debug=True)
