import pandas as pd
errors = pd.read_csv("reports/misclassified_samples.csv")
high_conf_errors = errors[
    ((errors["Predicted_Label"] == 1) & (errors["AI_Probability"] > 0.8)) |
    ((errors["Predicted_Label"] == 0) & (errors["AI_Probability"] < 0.2))
]
print("High-confidence misclassifications:", len(high_conf_errors))
high_conf_errors.to_csv(
    "reports/high_confidence_errors.csv",
    index=False
)
