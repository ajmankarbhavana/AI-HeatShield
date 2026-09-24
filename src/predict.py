
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Load trained Random Forest model
# --------------------------------------------------

model_path = "models/random_forest_heatshield.pkl"

model = joblib.load(model_path)

print("=" * 60)
print("AI HEATSHIELD - HEAT RISK PREDICTION")
print("=" * 60)

# --------------------------------------------------
# 2. Get environmental input from user
# --------------------------------------------------

print("\nEnter environmental information:")
print("(Press Enter to use the example values.)")

def get_number(prompt, default):
    value = input(f"{prompt} [{default}]: ")

    if value.strip() == "":
        return default

    try:
        return float(value)
    except ValueError:
        print("Invalid value. Using default value.")
        return default


temperature = get_number(
    "Temperature (°C)",
    42.0
)

humidity = get_number(
    "Humidity (%)",
    40.0
)

lst = get_number(
    "Land Surface Temperature - LST (°C)",
    45.0
)

ndvi = get_number(
    "NDVI",
    0.20
)

ndwi = get_number(
    "NDWI",
    0.10
)

ndbi = get_number(
    "NDBI",
    0.75
)

built_up = get_number(
    "Built-up area (%)",
    80.0
)

vegetation = get_number(
    "Vegetation area (%)",
    15.0
)

# --------------------------------------------------
# 3. Calculate derived features
# --------------------------------------------------

temperature_anomaly = temperature - 35

built_vegetation_ratio = (
    built_up / (vegetation + 1)
)

cooling_potential = (
    0.6 * ndvi
    + 0.3 * ndwi
    + 0.1 * (1 - built_up / 100)
)

heat_pressure = (
    0.35 * temperature
    + 0.25 * ndbi * 50
    + 0.20 * built_up
    - 0.15 * ndvi * 50
    - 0.10 * ndwi * 50
)

# --------------------------------------------------
# 4. Create input dataframe
# --------------------------------------------------

input_data = pd.DataFrame([{
    "temperature": temperature,
    "humidity": humidity,
    "lst": lst,
    "ndvi": ndvi,
    "ndwi": ndwi,
    "ndbi": ndbi,
    "built_up": built_up,
    "vegetation": vegetation,
    "temperature_anomaly": temperature_anomaly,
    "built_vegetation_ratio": built_vegetation_ratio,
    "cooling_potential": cooling_potential,
    "heat_pressure": heat_pressure
}])

# --------------------------------------------------
# 5. Make prediction
# --------------------------------------------------

prediction = model.predict(input_data)[0]

# --------------------------------------------------
# 6. Get prediction probabilities
# --------------------------------------------------

probabilities = model.predict_proba(input_data)[0]

classes = model.classes_

probability_table = pd.DataFrame({
    "Risk": classes,
    "Probability": probabilities
})

probability_table = probability_table.sort_values(
    by="Probability",
    ascending=False
)

# --------------------------------------------------
# 7. Display result
# --------------------------------------------------

print("\n" + "=" * 60)
print("AI HEATSHIELD PREDICTION")
print("=" * 60)

print(f"\nPredicted Heat Risk: {prediction.upper()}")

print("\nPrediction probabilities:")

for _, row in probability_table.iterrows():

    print(
        f"{row['Risk']:8s}: "
        f"{row['Probability'] * 100:.2f}%"
    )

# --------------------------------------------------
# 8. Display calculated features
# --------------------------------------------------

print("\nDerived environmental indicators:")

print(
    f"Temperature anomaly: "
    f"{temperature_anomaly:.2f} °C"
)

print(
    f"Built/vegetation ratio: "
    f"{built_vegetation_ratio:.2f}"
)

print(
    f"Cooling potential: "
    f"{cooling_potential:.3f}"
)

print(
    f"Heat pressure: "
    f"{heat_pressure:.2f}"
)

# --------------------------------------------------
# 9. Generate basic mitigation recommendations
# --------------------------------------------------

print("\nRecommended mitigation strategies:")

if prediction == "High":

    print("• Increase urban vegetation and tree cover.")
    print("• Increase shaded pedestrian and public areas.")
    print("• Consider cool or reflective surface strategies.")
    print("• Protect and restore water-sensitive areas.")
    print("• Prioritize high-heat zones for intervention.")

elif prediction == "Medium":

    print("• Increase vegetation where possible.")
    print("• Monitor changes in surface temperature.")
    print("• Improve shaded areas.")
    print("• Protect existing green and water areas.")

else:

    print("• Maintain existing vegetation.")
    print("• Continue monitoring heat conditions.")
    print("• Protect existing green and water areas.")
    print("• Avoid unnecessary increases in built-up surfaces.")

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

