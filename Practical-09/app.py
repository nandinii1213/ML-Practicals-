from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        feature_names = [
            "Variance",
            "Skewness",
            "Curtosis",
            "Entropy"
        ]

        # Read and validate all four feature values
        values = []

        for name in feature_names:
            value = float(data[name])

            if not pd.notna(value):
                raise ValueError("All feature values must be finite numbers.")

            import math
            if not math.isfinite(value):
                raise ValueError("All feature values must be finite numbers.")

            values.append(value)

        input_data = pd.DataFrame(
            [values],
            columns=feature_names
        )

        # Apply the same scaling used during training
        input_scaled = scaler.transform(input_data)

        prediction = int(model.predict(input_scaled)[0])

        if prediction == 0:
            result = "Genuine Banknote"
        else:
            result = "Counterfeit Banknote"

        return jsonify({
            "prediction": result,
            "class": prediction
        })

    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Please enter valid numeric values for all four features."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Please check the server setup."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)