
import os
import pandas as pd
import joblib

from xgboost import XGBClassifier

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
print("AI HEATSHIELD - XGBOOST MODEL")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

# --------------------------------------------------
# 2. Convert class labels into numbers
# --------------------------------------------------
# XGBoost works with numeric class labels.

label_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y_train_encoded = y_train.map(label_mapping)
y_test_encoded = y_test.map(label_mapping)

# --------------------------------------------------
# 3. Create XGBoost model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# --------------------------------------------------
# 4. Train the model
# --------------------------------------------------

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train_encoded
)

print("Model training completed!")

# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

y_pred_encoded = model.predict(X_test)

# Convert numeric predictions back to labels

reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

y_pred = pd.Series(y_pred_encoded).map(reverse_mapping)

# --------------------------------------------------
# 6. Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# --------------------------------------------------
# 7. Classification report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["Low", "Medium", "High"]
    )
)

# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low", "Medium", "High"]
)

print(cm)

# --------------------------------------------------
# 9. Feature importance
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
# 10. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_path = "models/xgboost_heatshield.json"

model.save_model(model_path)

print("\nModel saved successfully!")

print(f"Model path: {model_path}")

# --------------------------------------------------
# 11. Save feature importance
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

feature_importance.to_csv(
    "reports/xgboost_feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")

print("reports/xgboost_feature_importance.csv")

print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETED")
print("=" * 60)

