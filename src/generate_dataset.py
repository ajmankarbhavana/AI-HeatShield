
import os
import numpy as np
import pandas as pd

# Make results reproducible
np.random.seed(42)

# Number of locations/records
N = 1000

# --------------------------------------------------
# 1. Generate basic environmental variables
# --------------------------------------------------

temperature = np.random.normal(37, 4, N).clip(25, 48)

humidity = np.random.normal(55, 15, N).clip(20, 90)

# --------------------------------------------------
# 2. Generate satellite-style environmental indices
# --------------------------------------------------

ndvi = np.random.uniform(0.05, 0.85, N)

ndwi = np.random.uniform(0.00, 0.70, N)

ndbi = np.random.uniform(0.05, 0.90, N)

# --------------------------------------------------
# 3. Generate urban characteristics
# --------------------------------------------------

built_up = np.random.uniform(10, 95, N)

vegetation = np.random.uniform(5, 90, N)

# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------

# Temperature anomaly relative to a reference temperature
temperature_anomaly = temperature - 35

# Ratio between built-up area and vegetation
built_vegetation_ratio = built_up / (vegetation + 1)

# Cooling potential
cooling_potential = (
    0.6 * ndvi
    + 0.3 * ndwi
    + 0.1 * (1 - built_up / 100)
)

# Heat pressure
heat_pressure = (
    0.35 * temperature
    + 0.25 * ndbi * 50
    + 0.20 * built_up
    - 0.15 * ndvi * 50
    - 0.10 * ndwi * 50
)

# --------------------------------------------------
# 5. Create heat-risk score
# --------------------------------------------------

risk_score = (
    0.35 * temperature
    + 0.25 * ndbi * 50
    + 0.20 * built_up
    + 0.10 * temperature_anomaly
    - 0.20 * ndvi * 50
    - 0.10 * ndwi * 50
)

# Add a small amount of random variation
risk_score = risk_score + np.random.normal(0, 3, N)

# --------------------------------------------------
# 6. Convert risk score into categories
# --------------------------------------------------

high_threshold = np.percentile(risk_score, 70)
medium_threshold = np.percentile(risk_score, 35)

heat_risk = np.where(
    risk_score >= high_threshold,
    "High",
    np.where(
        risk_score >= medium_threshold,
        "Medium",
        "Low"
    )
)

# --------------------------------------------------
# 7. Create the final dataset
# --------------------------------------------------

df = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "lst": temperature + np.random.normal(1, 1.5, N),
    "ndvi": ndvi,
    "ndwi": ndwi,
    "ndbi": ndbi,
    "built_up": built_up,
    "vegetation": vegetation,
    "temperature_anomaly": temperature_anomaly,
    "built_vegetation_ratio": built_vegetation_ratio,
    "cooling_potential": cooling_potential,
    "heat_pressure": heat_pressure,
    "heat_risk": heat_risk
})

# --------------------------------------------------
# 8. Save dataset
# --------------------------------------------------

output_path = "data/raw/urban_heat_dataset.csv"

# Make sure the folder exists
os.makedirs("data/raw", exist_ok=True)

# Save CSV
df.to_csv(output_path, index=False)

# --------------------------------------------------
# 9. Display information
# --------------------------------------------------

print("=" * 50)
print("AI HEATSHIELD DATASET GENERATOR")
print("=" * 50)

print()

print("Dataset created successfully!")

print(f"Rows: {len(df)}")

print(f"Columns: {len(df.columns)}")

print()

print("Column names:")
print(df.columns.tolist())

print()

print("Risk distribution:")
print(df["heat_risk"].value_counts())

print()

print("First 5 records:")
print(df.head())

print()

print(f"Dataset saved to:")
print(output_path)

print()

print("=" * 50)
print("DATASET GENERATION COMPLETED")
print("=" * 50)

