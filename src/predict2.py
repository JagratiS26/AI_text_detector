import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from feature_engine2 import FeatureEngine
MODEL_PATH = "models/random_forest_model.pkl"

FEATURES = [
    'avg_sent_len',
    'vocab_richness',
    'burstiness',
    'readability',
    'avg_word_len',
    'punct_ratio',
    'stopword_ratio'
]

FIG_DIR = "figures/prediction_explanations"
os.makedirs(FIG_DIR, exist_ok=True)
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
def confidence_label(prob):
    if prob < 0.35:
        return "Confident Human"
    elif prob < 0.45:
        return "Borderline Human"
    elif prob < 0.55:
        return "Uncertain (Mixed Traits)"
    elif prob < 0.65:
        return "Borderline AI"
    else:
        return "Confident AI"

def predict_with_explanation(text):
    engine = FeatureEngine(text)
    features_dict = engine.extract_all()

    X = pd.DataFrame([features_dict], columns=FEATURES)

    prob_ai = model.predict_proba(X)[0][1]
    label = "AI" if prob_ai >= 0.5 else "Human"

    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   

    shap_vals = np.array(shap_vals).reshape(-1)

    n = min(len(FEATURES), len(shap_vals))

    shap_vals = shap_vals[:n]
    feature_names = FEATURES[:n]

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_vals
    }).sort_values(by="SHAP Value", key=abs)

    plt.figure(figsize=(7, 4))

    colors = ['#d62728' if v > 0 else '#2ca02c' for v in shap_df["SHAP Value"]]

    plt.barh(
        shap_df["Feature"],
        shap_df["SHAP Value"],
        color=colors
    )

    plt.axvline(0, color='black', linewidth=0.8)

    plt.title(f"Prediction: {label} | AI Probability: {prob_ai:.2f}")
    plt.xlabel("Impact on Prediction  → AI | ← Human")

    plt.tight_layout()

    filename = f"shap_{label}_{prob_ai:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(FIG_DIR, filename)

    plt.savefig(path, dpi=200)
    plt.close()

    return label, prob_ai, confidence_label(prob_ai), path

if __name__ == "__main__":
    text = input("\nEnter text:\n> ")

    label, prob, confidence, path = predict_with_explanation(text)

    print("\n====== RESULT ======")
    print("Label      :", label)
    print("AI Prob    :", round(prob, 2))
    print("Confidence :", confidence)
    print("Explanation Plot Saved At:")
    print(path)
