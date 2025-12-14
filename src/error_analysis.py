import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
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
    print(f" Removed {nan_count} rows with invalid labels")

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

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

error_df = X_test.copy()
error_df["True_Label"] = y_test.values
error_df["Predicted_Label"] = y_pred
error_df["AI_Probability"] = y_prob

errors = error_df[error_df["True_Label"] != error_df["Predicted_Label"]]

print("\nTotal test samples :", len(X_test))
print("Misclassified      :", len(errors))

print("\nError breakdown (True → Predicted):")
print(errors.groupby(["True_Label", "Predicted_Label"]).size())
errors.to_csv("reports/misclassified_samples.csv", index=False)
print("\nSaved: reports/misclassified_samples.csv")
