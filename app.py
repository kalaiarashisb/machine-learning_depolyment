import flask
from flask import request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = flask.Flask(__name__)

# Load trained model
try:
    model = joblib.load("decision_tree_model.joblib")
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found.")
    model = None


# Home Route
@app.route("/")
def home():
    return jsonify({
        "message": "Titanic Survival Prediction API is running!"
    })


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    # Check model loaded or not
    if model is None:
        return jsonify({
            "error": "Model not loaded"
        }), 500

    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Required features
        required_features = ["sex", "age", "class"]

        # Check missing features
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing feature: {feature}"
                }), 400

        # Create dataframe
        input_df = pd.DataFrame([{
            "sex": data["sex"],
            "age": data["age"],
            "class": data["class"]
        }])

        # Prediction
        prediction = model.predict(input_df)[0]

        # Convert prediction into readable format
        result = "Survived" if prediction == 1 else "Not Survived"

        # Return response
        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Run Flask App
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
