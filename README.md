# Carbon Footprint Forecaster

## Overview

This project provides a robust, high-accuracy predictive model to estimate an individual's **Carbon Footprint** (total greenhouse gas emissions expressed as CO₂ equivalent) based on daily behavioral data.

The goal was to transition from complex, slow estimation methods to a **real-time prediction tool** hosted online, empowering users to understand their environmental impact and make actionable, sustainable choices.

---

## Key Features & Technical Highlights

| Feature | Description | 
| :----- | :----- | 
| **High Accuracy** | Achieved a state-of-the-art predictive performance with a **Mean Absolute Error (MAE) of ~64** on the test set. | 
| **Model Optimization** | Comparative analysis across 7+ models (Linear, Elastic Net, XGBoost, etc.) led to the selection of a highly interpretable **Linear Model with Pairwise Interactions** as the optimal predictor. | 
| **Advanced EDA** | Utilized R's **Factor Analysis of Mixed Data (FAMD)** and **Multi-Factor Analysis (MFA)** to uncover non-linear relationships and prioritize the most influential features. | 
| **Data-Driven Insights** | Identified **Transport, Air Travel Frequency, and Vehicle Type** as the strongest drivers of carbon emissions, guiding model feature engineering. | 
| **Deployment** | Implemented the final model via a **Streamlit application** for a user-friendly, real-time prediction interface (live deployment link available in repository). | 

---

## Technologies Used

The entire data preparation and modeling pipeline was executed within the R ecosystem, with deployment handled via Python.

### **R/RStudio (Modeling & Analytics)**

* `caret` / `glmnet`: Model training, cross-validation, and Elastic Net implementation.

* `FactoMineR`: Execution of FAMD and MFA for dimension reduction and categorical variable analysis.

* `tidyverse` (e.g., `ggplot2`): Data cleaning, preprocessing, and generating the exploratory plots (Partial Dependence Plots, Residual Plots).

### **Python (Deployment)**

* `Streamlit`: Web application framework for the real-time prediction interface.

---

## Methodology and Modeling Insights

### 1. Data Analysis (R, FAMD, & MFA)

Rigorous Exploratory Data Analysis was performed on a synthesized dataset (10,000 rows, 20 features) covering aspects like diet, waste, and distance traveled.

* **Key Findings:** The analysis confirmed that **Transport Type** and **Distance Travelled Per Month** are the largest contributors to overall variance and carbon output.

* **Interaction Term Importance:** Plot analysis revealed the non-additive effect of combining features (e.g., using a **private petrol vehicle** has a significantly higher emission impact than the sum of its parts).

### 2. Model Training & Selection

A series of linear and non-linear models were tested to find the best balance between predictive power and interpretability.

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | Notes | 
| :----- | :----- | :----- | :----- | 
| **LM w/ Pairwise Interactions** | **~64** | **~100** | **Selected Model (Best Performer)**. Highly accurate and offers clear coefficients for auditing. | 
| XGBoost (Tuned) | ~110 | ~190 | Strong performance, but less interpretable than the final choice. | 
| Neural Network (Tuned) | ~125 | ~220 | Overly complex for the required accuracy gain. | 

### 3. Feature Importance (Elastic Net)

Elastic Net regularization was used to confirm feature importance and prevent overfitting. The analysis consistently prioritized interaction terms:

* `Dist_TravelledPM:Vehicle_Typeelectric`

* `AirTravel_Freqvery frequently`

* `Body_Typeobese:Vehicle_Typeelectric`

---

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/sai-vivek108/Carbon-Foot-Print-Forecaster.git](https://github.com/sai-vivek108/Carbon-Foot-Print-Forecaster.git)
