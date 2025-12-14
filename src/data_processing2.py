import pandas as pd
data = pd.read_csv('data/final_high_quality_features_v2.csv')
print("Dataset loaded. Shape:", data.shape)
features = [
    'avg_sent_len',
    'vocab_richness',
    'burstiness',
    'readability',
    'avg_word_len',
    'punct_ratio',
    'stopword_ratio'
]
print("\nCleaning labels...")

data['label'] = pd.to_numeric(data['label'], errors='coerce')
invalid = data['label'].isna().sum()

if invalid > 0:
    print(f" Removed {invalid} invalid label rows")

data = data.dropna(subset=['label']).reset_index(drop=True)
data['label'] = data['label'].astype(int)
X = data[features]
y = data['label']

print("\nFeatures:", features)
print("\nTarget distribution:\n", y.value_counts())
print("\nFinal shape:", data.shape)
