
import os
import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 70)
print("AI HEATSHIELD - MODEL COMPARISON")
print("=" * 70)

# --------------------------------------------------
# 1. Load test data
# --------------------------------------------------

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("\nTest data:")
print(f"Samples: {len(X_test)}")
print(f"Features: {X_test.shape[1]}")

# --------------------------------------------------
# 2. Load Random Forest model
# --------------------------------------------------

print("\nLoading Random Forest model...")

rf_model = joblib.load(
    "models/random_forest_heatshield.pkl"
)

# Random Forest predictions
rf_predictions = rf_model.predict(X_test)

# --------------------------------------------------
# 3. Calculate Random Forest metrics
# --------------------------------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="macro"
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="macro"
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="macro"
)

# --------------------------------------------------
# 4. Load XGBoost model
# --------------------------------------------------

print("Loading XGBoost model...")

xgb_model = XGBClassifier()

xgb_model.load_model(
    "models/xgboost_heatshield.json"
)

# XGBoost predictions
xgb_predictions_numeric = xgb_model.predict(X_test)

# Convert numeric predictions back to labels
reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

xgb_predictions = pd.Series(
    xgb_predictions_numeric
).map(reverse_mapping)

# --------------------------------------------------
# 5. Calculate XGBoost metrics
# --------------------------------------------------

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

xgb_precision = precision_score(
    y_test,
    xgb_predictions,
    average="macro"
)

xgb_recall = recall_score(
    y_test,
    xgb_predictions,
    average="macro"
)

xgb_f1 = f1_score(
    y_test,
    xgb_predictions,
    average="macro"
)

# --------------------------------------------------
# 6. Create comparison table
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],
    "Precision": [
        rf_precision,
        xgb_precision
    ],
    "Recall": [
        rf_recall,
        xgb_recall
    ],
    "F1_Score": [
        rf_f1,
        xgb_f1
    ]
})

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1_Score": "{:.4f}".format
        }
    )
)

# --------------------------------------------------
# 8. Convert metrics to percentages
# --------------------------------------------------

print("\nPerformance percentages:")

for _, row in comparison.iterrows():

    print(f"\n{row['Model']}")

    print(
        f"Accuracy : {row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: {row['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : {row['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : {row['F1_Score'] * 100:.2f}%"
    )

# --------------------------------------------------
# 9. Identify higher measured metrics
# --------------------------------------------------

best_accuracy_model = comparison.loc[
    comparison["Accuracy"].idxmax(),
    "Model"
]

best_f1_model = comparison.loc[
    comparison["F1_Score"].idxmax(),
    "Model"
]

print("\n" + "=" * 70)

print(
    f"Highest measured accuracy on this test set: "
    f"{best_accuracy_model}"
)

print(
    f"Highest measured macro F1 on this test set: "
    f"{best_f1_model}"
)

print("=" * 70)

# --------------------------------------------------
# 10. Save comparison results
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")

print("reports/model_comparison.csv")

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

