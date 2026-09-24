
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI HeatShield",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/random_forest_heatshield.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Random Forest model not found. "
        "Please train the model first."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-high {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        background-color: #5c1f1f;
    }

    .risk-medium {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        background-color: #66551a;
    }

    .risk-low {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        background-color: #174d32;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 AI HeatShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Urban Heat Risk Prediction and Mitigation System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Environmental Inputs")

st.sidebar.info(
    "Enter environmental and urban characteristics "
    "to estimate the heat-risk category."
)


temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=60.0,
    value=42.0,
    step=0.1
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

lst = st.sidebar.number_input(
    "Land Surface Temperature - LST (°C)",
    min_value=0.0,
    max_value=70.0,
    value=45.0,
    step=0.1
)

ndvi = st.sidebar.number_input(
    "NDVI",
    min_value=-1.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

ndwi = st.sidebar.number_input(
    "NDWI",
    min_value=-1.0,
    max_value=1.0,
    value=0.10,
    step=0.01
)

ndbi = st.sidebar.number_input(
    "NDBI",
    min_value=-1.0,
    max_value=1.0,
    value=0.75,
    step=0.01
)

built_up = st.sidebar.number_input(
    "Built-up Area (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=0.1
)

vegetation = st.sidebar.number_input(
    "Vegetation Area (%)",
    min_value=0.0,
    max_value=100.0,
    value=15.0,
    step=0.1
)


# ============================================================
# CALCULATE DERIVED FEATURES
# ============================================================

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


# ============================================================
# CREATE MODEL INPUT
# ============================================================

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


# ============================================================
# MAIN DASHBOARD
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🌡️ Environmental Conditions")

    st.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )

    st.metric(
        "Humidity",
        f"{humidity:.1f} %"
    )

    st.metric(
        "Land Surface Temperature",
        f"{lst:.1f} °C"
    )


with col2:

    st.subheader("🌿 Urban Indicators")

    st.metric(
        "NDVI",
        f"{ndvi:.2f}"
    )

    st.metric(
        "NDBI",
        f"{ndbi:.2f}"
    )

    st.metric(
        "Vegetation",
        f"{vegetation:.1f} %"
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Heat Risk",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

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


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔥 AI Heat-Risk Assessment")

    if prediction == "High":

        st.markdown(
            '<div class="risk-high">'
            '🔥 HIGH HEAT RISK'
            '</div>',
            unsafe_allow_html=True
        )

    elif prediction == "Medium":

        st.markdown(
            '<div class="risk-medium">'
            '⚠️ MEDIUM HEAT RISK'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="risk-low">'
            '🌿 LOW HEAT RISK'
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.subheader("📊 Prediction Confidence")

    probability_display = probability_table.copy()

    probability_display["Probability"] = (
        probability_display["Probability"] * 100
    ).round(2)

    probability_display["Probability"] = (
        probability_display["Probability"].astype(str)
        + "%"
    )

    st.dataframe(
        probability_display,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Derived indicators
    # --------------------------------------------------------

    st.subheader("🧮 Derived Environmental Indicators")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Temperature Anomaly",
            f"{temperature_anomaly:.2f} °C"
        )

    with metric2:
        st.metric(
            "Built/Vegetation Ratio",
            f"{built_vegetation_ratio:.2f}"
        )

    with metric3:
        st.metric(
            "Cooling Potential",
            f"{cooling_potential:.3f}"
        )

    with metric4:
        st.metric(
            "Heat Pressure",
            f"{heat_pressure:.2f}"
        )


    # --------------------------------------------------------
    # Probability chart
    # --------------------------------------------------------

    st.subheader("📈 Risk Probability Distribution")

    chart_data = probability_table.set_index("Risk")

    st.bar_chart(
        chart_data["Probability"]
    )


    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    st.subheader("🤖 Model Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": input_data.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # Feature importance chart
    # --------------------------------------------------------

    fig, ax = plt.subplots()

    top_features = importance_df.head(8)

    ax.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    ax.set_xlabel("Importance")

    ax.set_title(
        "Top Model Features"
    )

    st.pyplot(fig)


    # --------------------------------------------------------
    # Mitigation recommendations
    # --------------------------------------------------------

    st.subheader("🌱 Recommended Mitigation Strategies")

    if prediction == "High":

        recommendations = [
            "Increase urban vegetation and tree cover.",
            "Increase shaded pedestrian and public areas.",
            "Consider cool or reflective surface strategies.",
            "Protect and restore water-sensitive areas.",
            "Prioritize high-heat zones for intervention."
        ]

    elif prediction == "Medium":

        recommendations = [
            "Increase vegetation where possible.",
            "Monitor changes in surface temperature.",
            "Improve shaded areas.",
            "Protect existing green and water areas."
        ]

    else:

        recommendations = [
            "Maintain existing vegetation.",
            "Continue monitoring heat conditions.",
            "Protect existing green and water areas.",
            "Avoid unnecessary increases in built-up surfaces."
        ]


    for recommendation in recommendations:

        st.write(
            "• " + recommendation
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI HeatShield | Urban Heat Risk Prediction Prototype"
)

st.caption(
    "Model: Random Forest | "
    "Data: Synthetic development dataset"
)

