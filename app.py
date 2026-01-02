from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Import your prediction logic
from src.predict2 import predict_with_explanation

app = Flask(__name__)
CORS(app)

# Path for SHAP images
SHAP_DIR = "figures/prediction_explanations"

@app.route("/")
def home():
    return "AI Text Detector Backend is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Call your ML + SHAP pipeline
        label, ai_prob, confidence, shap_path = predict_with_explanation(text)

        return jsonify({
            "label": label,
            "ai_probability": round(ai_prob, 2),
            "confidence": confidence,
            "shap_plot": shap_path
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


# Serve SHAP images
@app.route("/figures/prediction_explanations/<path:filename>")
def serve_shap_image(filename):
    return send_from_directory(SHAP_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
