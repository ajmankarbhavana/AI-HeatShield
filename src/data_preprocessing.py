import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

input_path = "data/raw/urban_heat_dataset.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("AI HEATSHIELD - DATA PREPROCESSING")
print("=" * 60)

print("\nDataset loaded successfully!")

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# --------------------------------------------------
# 2. Display dataset information
# --------------------------------------------------

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

# --------------------------------------------------
# 4. Check duplicate rows
# --------------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate rows: {duplicates}")

# Remove duplicates if any exist
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

# --------------------------------------------------
# 5. Statistical summary
# --------------------------------------------------

print("\nStatistical summary:")
print(df.describe())

# --------------------------------------------------
# 6. Check target distribution
# --------------------------------------------------

print("\nHeat-risk distribution:")
print(df["heat_risk"].value_counts())

# --------------------------------------------------
# 7. Separate features and target
# --------------------------------------------------

X = df.drop("heat_risk", axis=1)

y = df["heat_risk"]

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --------------------------------------------------
# 9. Create processed directory
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)

# --------------------------------------------------
# 10. Save processed datasets
# --------------------------------------------------

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("\nProcessed datasets saved successfully!")

print("\nFiles created:")
print("data/processed/X_train.csv")
print("data/processed/X_test.csv")
print("data/processed/y_train.csv")
print("data/processed/y_test.csv")

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED")
print("=" * 60)