# 🌍 AI HeatShield

## AI-Based Urban Heat Risk Prediction and Mitigation System

AI HeatShield is a machine-learning-based prototype designed to predict urban heat-risk levels using environmental and urban indicators and provide mitigation recommendations.

The system uses **Random Forest** and **XGBoost** classification models and provides an interactive **Streamlit dashboard** for heat-risk prediction.

---

## 🎯 Objectives

* Predict urban heat-risk levels as **Low, Medium, or High**.
* Process environmental and urban characteristics.
* Compare different machine-learning models.
* Identify important factors contributing to heat-risk predictions.
* Provide mitigation recommendations based on the predicted risk.
* Demonstrate an end-to-end AI data pipeline from data generation to deployment.

---

## 🏗️ System Architecture

```text
Environmental Data
        │
        ▼
Data Generation / Acquisition
        │
        ▼
Data Preprocessing
        │
        ├── Missing-value checking
        ├── Duplicate checking
        ├── Feature preparation
        └── Train/Test Split
        │
        ▼
Feature Engineering
        │
        ├── Temperature Anomaly
        ├── Built/Veg. Ratio
        ├── Cooling Potential
        └── Heat Pressure
        │
        ▼
Machine Learning Models
        │
        ├── Random Forest
        │
        └── XGBoost
        │
        ▼
Model Comparison
        │
        ▼
Selected Prediction Model
        │
        ▼
Heat-Risk Prediction
        │
        ├── Low
        ├── Medium
        └── High
        │
        ▼
Mitigation Recommendations
        │
        ▼
Streamlit Dashboard
```

---

## 🤖 Machine Learning Models

### Random Forest

Random Forest is used as the primary classification model for the current prototype.

Measured performance on the current test dataset:

* Accuracy: **76.00%**
* Precision: **76.16%**
* Recall: **76.35%**
* F1 Score: **76.24%**

### XGBoost

XGBoost is implemented as a second classification model for comparison.

Measured performance on the current test dataset:

* Accuracy: **74.00%**
* Precision: **73.97%**
* Recall: **74.21%**
* F1 Score: **74.08%**

The comparison is based on the current synthetic development dataset and test split.

---

## 📊 Dataset

The current prototype uses a **synthetic development dataset containing 1,000 records**.

The dataset contains environmental and urban features including:

* Temperature
* Humidity
* Land Surface Temperature (LST)
* NDVI
* NDWI
* NDBI
* Built-up area
* Vegetation
* Temperature anomaly
* Built/vegetation ratio
* Cooling potential
* Heat pressure
* Heat-risk label

### Target Classes

```text
Low
Medium
High
```

---

## 🧮 Feature Engineering

The system derives additional indicators from the input variables.

### Temperature Anomaly

Represents the difference between the input temperature and the reference temperature used by the prototype.

### Built/Veg. Ratio

Represents the relationship between built-up area and vegetation area.

### Cooling Potential

Combines vegetation, water-related information, and built-up characteristics.

### Heat Pressure

Combines several environmental and urban indicators into a derived heat-pressure measure.

---

## 🔄 Data Processing Pipeline

```text
Raw Dataset
     │
     ▼
Data Quality Checks
     │
     ├── Missing Values
     └── Duplicate Records
     │
     ▼
Feature Matrix
     │
     ▼
Train/Test Split
     │
     ├── Training Data: 800 samples
     └── Testing Data: 200 samples
     │
     ▼
Model Training
     │
     ├── Random Forest
     └── XGBoost
     │
     ▼
Evaluation
     │
     ▼
Prediction
```

---

## 🔥 Prediction System

The prediction module accepts environmental inputs and calculates derived indicators before sending the feature vector to the trained Random Forest model.

The system returns:

* Predicted heat-risk category
* Prediction probabilities
* Derived environmental indicators
* Mitigation recommendations

Example output:

```text
Predicted Heat Risk: HIGH

Temperature anomaly: 7.00 °C
Built/vegetation ratio: 5.00
Cooling potential: 0.170
Heat pressure: 38.08
```

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard provides:

* Environmental input controls
* Urban indicators
* Heat-risk prediction
* Prediction confidence
* Derived environmental indicators
* Risk probability distribution
* Feature importance
* Mitigation recommendations

Run the dashboard with:

```bash
python -m streamlit run app/app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
AI-HeatShield/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── urban_heat_dataset.csv
│   │
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── random_forest_heatshield.pkl
│   └── xgboost_heatshield.json
│
├── notebooks/
│
├── reports/
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── xgboost_feature_importance.csv
│
├── src/
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── train_xgboost.py
│   ├── model_comparison.py
│   └── predict.py
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ajawankarbhavanasri-sketch/AI-HeatShield.git
```

Move into the project directory:

```bash
cd AI-HeatShield
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Generate the dataset

```bash
python src/generate_dataset.py
```

### Preprocess the dataset

```bash
python src/data_preprocessing.py
```

### Train Random Forest

```bash
python src/train_model.py
```

### Train XGBoost

```bash
python src/train_xgboost.py
```

### Compare models

```bash
python src/model_comparison.py
```

### Run command-line prediction

```bash
python src/predict.py
```

### Launch the Streamlit dashboard

```bash
python -m streamlit run app/app.py
```

---

## 📈 Current Results

| Model         | Accuracy | Precision | Recall | F1 Score |
| ------------- | -------: | --------: | -----: | -------: |
| Random Forest |   76.00% |    76.16% | 76.35% |   76.24% |
| XGBoost       |   74.00% |    73.97% | 74.21% |   74.08% |

These measurements are from the current **synthetic development dataset** and its 200-sample test set.

---

## ⚠️ Current Limitations

* The current dataset is synthetic and intended for development/prototyping.
* The model has not been validated against real-world urban heat observations.
* Satellite imagery and live environmental data are not yet integrated.
* The current dashboard is a prototype prediction interface.
* Predictions should not be interpreted as operational heat-risk assessments.

---

## 🚀 Future Scope

Future versions can integrate:

* Real satellite imagery
* Landsat and Sentinel-derived indices
* Geographic Information System (GIS) data
* Real-time weather information
* Urban land-use data
* Spatial heat-risk mapping
* Location-based prediction
* More advanced machine-learning models
* Automated model retraining
* Real-world validation
* Cloud deployment

---

## 🛠️ Technology Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Matplotlib**
* **Streamlit**
* **Git & GitHub**

---

## 📌 Project Status

**Current Status: Working Prototype**

The current implementation includes dataset generation, preprocessing, machine-learning model training, model comparison, command-line prediction, and an interactive Streamlit dashboard.

---

## 👩‍💻 Author

**Ajmankar Bhavana**

AI HeatShield — AI-Based Urban Heat Risk Prediction and Mitigation System
