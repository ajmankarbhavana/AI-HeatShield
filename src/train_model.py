
import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# 1. Load processed training and testing data
# --------------------------------------------------

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("=" * 60)
print("AI HEATSHIELD - RANDOM FOREST MODEL")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# --------------------------------------------------
# 2. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

# --------------------------------------------------
# 3. Train the model
# --------------------------------------------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")

# --------------------------------------------------
# 4. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# --------------------------------------------------
# 5. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# --------------------------------------------------
# 6. Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print(cm)

# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# --------------------------------------------------
# 9. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/random_forest_heatshield.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(f"Model path: {model_path}")

# --------------------------------------------------
# 10. Save feature importance
# --------------------------------------------------

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

print("Feature importance saved to:")
print("reports/feature_importance.csv")

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 60)

