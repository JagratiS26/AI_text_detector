import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

DATA_PATH = "data/final_high_quality_features_v2.csv"
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

data = pd.read_csv(DATA_PATH)
print("Original data shape:", data.shape)
data['label'] = pd.to_numeric(data['label'], errors='coerce')

nan_count = data['label'].isna().sum()
if nan_count > 0:
    print(f"⚠ Removed {nan_count} rows with invalid labels")

data = data.dropna(subset=['label']).reset_index(drop=True)
data['label'] = data['label'].astype(int)

print("Cleaned data shape:", data.shape)
print("Label distribution:\n", data['label'].value_counts())

X = data[FEATURES]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = joblib.load(MODEL_PATH)
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (AI vs Human)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=200)
plt.show()
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("figures/precision_recall_curve.png", dpi=200)
plt.show()
