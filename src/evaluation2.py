import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

data = pd.read_csv('data/final_high_quality_features_v2.csv')
data['label'] = pd.to_numeric(data['label'], errors='coerce')
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)

features = [
    'avg_sent_len',
    'vocab_richness',
    'burstiness',
    'readability',
    'avg_word_len',
    'punct_ratio',
    'stopword_ratio'
]

X = data[features]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load("models/random_forest_model.pkl")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n")
print(classification_report(
    y_test, y_pred,
    target_names=["Human", "AI"]
))
