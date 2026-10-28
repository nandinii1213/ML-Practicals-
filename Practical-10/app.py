from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained SVM model and metadata
model = joblib.load("nonlinear_svm_model.pkl")
metadata = joblib.load("model_metadata.pkl")

features = metadata["features"]
defaults = metadata["defaults"]
target_names = metadata["target_names"]


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
        # Read the six input values
        values = [float(request.form[feature]) for feature in features]

        # Create input DataFrame in the correct feature order
        input_data = pd.DataFrame([values], columns=features)

        # Predict class using the trained model
        prediction = int(model.predict(input_data)[0])
        result = target_names[prediction]

        return render_template(
            "index.html",
            features=features,
            defaults=defaults,
            prediction=result
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            features=features,
            defaults=defaults,
            error="Please enter valid numerical values in all six fields."
        )


if __name__ == "__main__":
    app.run(debug=True)