from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained Naive Bayes model
# Make sure naive_bayes_model.pkl is in the SAME folder as this app.py
model = joblib.load("naive_bayes_model.pkl")

# Load the text vectorizer
# Make sure vectorizer.pkl is in the SAME folder as this app.py
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get the message entered by the user
        message = data.get("message", "").strip()

        # Check whether message is empty
        if not message:
            return jsonify({
                "error": "Please enter a message."
            }), 400

        # Convert text into Bag-of-Words using the saved vectorizer
        message_bow = vectorizer.transform([message])

        # Predict the category
        prediction = model.predict(message_bow)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(message_bow)[0]

        # Confidence
        confidence = round(float(max(probabilities)) * 100, 2)

        # Send result back to HTML
        result = {
            "prediction": str(prediction),
            "confidence": confidence
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)