import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    'avg_sent_len',
    'vocab_richness',
    'burstiness',
    'readability',
    'avg_word_len',
    'punct_ratio',
    'stopword_ratio'
]
df = pd.read_csv("reports/misclassified_samples.csv")
full = pd.read_csv("data/final_high_quality_features_v2.csv")

correct = full[FEATURES]
incorrect = df[FEATURES]

mean_correct = correct.mean()
mean_incorrect = incorrect.mean()

plt.figure(figsize=(7,4))
(mean_incorrect - mean_correct).plot(kind="bar")

plt.title("Feature Difference: Misclassified vs Correct")
plt.ylabel("Mean Difference")

plt.tight_layout()
plt.savefig("figures/error_feature_difference.png", dpi=200)
plt.show()
