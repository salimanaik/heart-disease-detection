# heart-disease-detection
 Heart Disease ML Prediction App

An end-to-end **Machine Learning–based Heart Disease Prediction System** with data analysis, multiple ML models, and an interactive **Streamlit frontend** for real-time risk prediction.

##  Project Overview

This project predicts the likelihood of heart disease using patient clinical data.  
It demonstrates the complete ML workflow — from **exploratory data analysis (EDA)** and **model training** to **model comparison** and a **user-facing frontend**.

##  Dataset Information

- **Total Records:** 400
- **Features:** 13 clinical attributes
- **Target Variable:** `heart_disease`
  - `0` → No Heart Disease  
  - `1` → Heart Disease

### Key Features
- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Max Heart Rate  
- ST Depression  
- Number of Major Vessels  
- Thalassemia  

##  Exploratory Data Analysis (EDA)

EDA was performed to understand feature distributions and relationships:
- Target class distribution
- Correlation heatmap of numerical features
- Scatter plots (e.g., Age vs Max Heart Rate)


##  Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

**Recall and F1-Score were prioritized** due to their importance in medical diagnosis.



## Model Training Pipeline

1. Data preprocessing and encoding  
2. Feature scaling using `StandardScaler`  
3. 80–20 train-test split  
4. Training multiple ML classifiers  
5. Model performance comparison  
6. Saving trained models and scaler using `joblib`
   
scaler.joblib
logistic_regression_model.joblib
decision_tree_model.joblib
random_forest_model.joblib
svm_model.joblib


## Frontend (Streamlit)

The Streamlit application provides:
- Dataset preview
- Feature distribution visualizations
- Model training and comparison
- Patient data input form
- Risk prediction from all models
- Consensus heart disease risk output

##  How to Run the Project

###  Install Dependencies
```bash
pip install -r requirements.txt


streamlit run app.py


### 📁 Saved Model Files
