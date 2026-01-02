from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend is running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # TEMP DUMMY RESPONSE (ML baad me add karenge)
    return jsonify({
        "label": "AI",
        "ai_probability": 0.61,
        "confidence": "Borderline AI",
        "shap_plot": None
    })

if __name__ == "__main__":
    app.run(debug=True)

