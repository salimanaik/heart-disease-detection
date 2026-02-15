import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Heart Disease Detection", page_icon="❤️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .model-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #d32f2f;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and return the heart disease dataset"""
    # Create sample data based on standard heart disease dataset structure
    np.random.seed(42)
    n_samples = 400
    
    data = {
        'age': np.random.randint(29, 71, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'cp': np.random.choice([0, 1, 2, 3], n_samples),
        'trestbps': np.random.normal(131, 17, n_samples).clip(94, 200),
        'chol': np.random.normal(246, 52, n_samples).clip(126, 564),
        'fbs': np.random.choice([0, 1], n_samples),
        'restecg': np.random.choice([0, 1, 2], n_samples),
        'thalach': np.random.normal(150, 23, n_samples).clip(71, 202),
        'exang': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples),
        'slope': np.random.choice([1, 2, 3], n_samples),
        'ca': np.random.choice([0, 1, 2, 3], n_samples),
        'thal': np.random.choice([0, 1, 2, 3], n_samples),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    }
    
    df = pd.DataFrame(data)
    return df

def train_models(X_train, y_train, X_test, y_test):
    """Train all models and return predictions and scores"""
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
        }
        predictions[name] = y_pred
        probabilities[name] = y_prob
    
    return results, predictions, probabilities

# Main app
def main():
    st.markdown('<h1 class="main-header">❤️ Heart Disease Detection</h1>', unsafe_allow_html=True)
    
    
    
    # Load data
    @st.cache_data
    def cached_load_data():
        return load_data()
    
    df = cached_load_data()
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df), delta="400 records")
    with col2:
        heart_disease_rate = df['heart_disease'].mean() * 100
        st.metric("Heart Disease Rate", f"{heart_disease_rate:.1f}%", delta="+2.5%")
    with col3:
        avg_age = df['age'].mean()
        st.metric("Avg Age", f"{avg_age:.0f}", delta="54 years")
    with col4:
        avg_chol = df['chol'].mean()
        st.metric("Avg Cholesterol", f"{avg_chol:.0f}", delta="246 mg/dl")
    
    # Data preview
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Feature distribution
    st.subheader("📈 Feature Distributions")
    selected_features = st.multiselect("Select features to visualize", 
                                     df.columns.tolist(), default=['age', 'chol', 'thalach', 'heart_disease'])
    
    if selected_features:
        cols = st.columns(2)
        for i, feature in enumerate(selected_features):
            with cols[i % 2]:
                fig, ax = plt.subplots()
                if df[feature].nunique() <= 10:
                    sns.countplot(data=df, x=feature, hue='heart_disease', ax=ax)
                else:
                    sns.histplot(data=df, x=feature, hue='heart_disease', kde=True, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    # Prepare data for modeling
    st.subheader("🤖 Model Training & Evaluation")
    
    feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    X = df[feature_cols]
    y = df['heart_disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training models..."):
            results, predictions, probabilities = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Model comparison table
        st.subheader("📊 Model Performance Comparison")
        
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] else 'N/A'
            })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        st.success(f"🏆 **Best Model: {best_model[0]}** (F1-Score: {best_model[1]['f1']:.3f})")
        
        # Model cards
        st.subheader("📈 Detailed Model Analysis")
        model_cols = st.columns(2)
        
        for i, (name, metrics) in enumerate(results.items()):
            with model_cols[i % 2]:
                st.markdown(f"""
                <div class="model-card">
                    <h3>{name}</h3>
                    <p><strong>Accuracy:</strong> {metrics['accuracy']:.3f}</p>
                    <p><strong>Precision:</strong> {metrics['precision']:.3f}</p>
                    <p><strong>Recall:</strong> {metrics['recall']:.3f}</p>
                    <p><strong>F1-Score:</strong> {metrics['f1']:.3f}</p>
                    <p><strong>ROC-AUC:</strong> {metrics['roc_auc']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.session_state.results = results
        st.session_state.predictions = predictions
        st.session_state.probabilities = probabilities
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_test = y_test
        st.rerun()
    
    # Prediction interface
    st.subheader("🔮 Predict Heart Disease Risk")
    st.sidebar.header("Model Selection")

    model_display = st.sidebar.selectbox(
            "Select Prediction Model",
            ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
        )

    model_map = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
            "SVM": "svm"
        }

    selected_model = model_map[model_display]

    
    if 'results' in st.session_state:
        # Patient input form
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 29, 71, 54)
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Angina", "Asymptomatic"][x])
                trestbps = st.slider("Resting BP (mm Hg)", 94, 200, 130)
            
            with col2:
                chol = st.slider("Cholesterol (mg/dl)", 126, 564, 246)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
                restecg = st.selectbox("Resting ECG", [0, 1, 2])
                thalach = st.slider("Max Heart Rate", 71, 202, 150)
            
            col3, col4 = st.columns(2)
            with col3:
                exang = st.selectbox("Exercise Induced Angina", [0, 1])
                oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
                slope = st.selectbox("ST Slope", [1, 2, 3])
            
            with col4:
                ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
                thal = st.selectbox("Thalassemia", [0, 1, 2, 3])
            
            predict_btn = st.form_submit_button("Predict Risk", type="primary")
        
            if predict_btn:

                API_URL = f"https://heart-disease-api-uixi.onrender.com/predict?model_name={selected_model}"

                payload = {
                    "age": age,
                    "sex": sex,
                    "chest_pain_type": cp,
                    "resting_blood_pressure": trestbps,
                    "cholesterol": chol,
                    "fasting_blood_sugar": fbs,
                    "resting_ecg": restecg,
                    "max_heart_rate": thalach,
                    "exercise_induced_angina": exang,
                    "st_depression": oldpeak,
                    "st_slope": slope,
                    "num_major_vessels": ca,
                    "thalassemia": thal
                }

                try:

                    response = requests.post(API_URL, json=payload, timeout=120)

                    if response.status_code == 200:

                        result = response.json()

                        prediction = result["prediction"]
                        probability = result["probability"]
                        interpretation = result["interpretation"]

                        st.info(f"Model Used: {model_display}")

                        if prediction == 1:
                            st.error(f"🔴 {interpretation}")
                        else:
                            st.success(f"🟢 {interpretation}")

                        st.metric("Probability", f"{probability:.2%}")

                    else:
                        st.error("API Error")

                except Exception as e:
                    st.error(f"Connection Error: {e}")


            
            # Consensus prediction
        consensus = sum(st.session_state.predictions[model_name][0] for model_name in st.session_state.predictions) / 4
        st.markdown(f"### 🎯 **Consensus Risk: {'🔴 HIGH' if consensus >= 0.5 else '🟢 LOW'}** ({consensus:.1%})")
        

    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Click "Train All Models"** to train and compare all 4 classification algorithms
        2. **Review model performance** metrics (Focus on Recall & F1-Score for medical applications)
        3. **Enter patient data** in the prediction form
        4. **Get instant risk assessment** from all models + consensus prediction
        
        **Key Metrics Explained:**
        - **Recall**: % of actual heart disease cases correctly identified (most important)
        - **Precision**: % of positive predictions that were actually positive
        - **F1-Score**: Balance between Precision and Recall
        - **ROC-AUC**: Model's ability to distinguish classes
        """)

if __name__ == "__main__":
    main()
