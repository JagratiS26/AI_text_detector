import pandas as pd
from feature_engine2 import FeatureEngine
data = pd.read_csv("data/dataset.csv")

print("Dataset loaded:", data.shape)

features_list = []

for text in data["text"]:
    engine = FeatureEngine(text)
    features_list.append(engine.extract_all())

features_df = pd.DataFrame(features_list)
features_df["label"] = data["label"].values
features_df.to_csv("data/final_high_quality_features_v2.csv", index=False)

print("Feature dataset rebuilt:", features_df.shape)
