import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_processing2 import X, y
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest_model.pkl")
print(" Model saved to models/random_forest_model.pkl")
