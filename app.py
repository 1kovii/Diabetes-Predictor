import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION AND DATA LOADING ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥", layout="wide")

# Function to load data directly from the CSV file
@st.cache_data
def load_data():
    try:
        # Assuming the CSV file is in the same directory as this script
        df = pd.read_csv('diabetes.csv')
        # Standardize column names if they are different (e.g., if 'Outcome' is 'Class')
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        return df
    except FileNotFoundError:
        st.error("ERROR: 'diabetes.csv' not found. Please ensure the file is in the same directory as app.py.")
        return pd.DataFrame()

# Data Cleaning and Imputation
@st.cache_data
def preprocess_data(df):
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Replace 0 values with NaN for specific columns, then fill with mean
    # These columns cannot logically be zero (except for Pregnancies/DPF/Outcome)
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_to_clean:
        # Replace 0s with NaN
        df_clean[col] = df_clean[col].replace(0, np.nan) 
        # Fill NaN values with the column mean (Imputation)
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    return df_clean

# Train the Machine Learning Model
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, 0, None, None, None, None

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions and calculate metrics
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, accuracy, conf_matrix, class_report, X_test, y_test

# Load and process data
data = load_data()
clean_data = preprocess_data(data)

# Check if data loaded successfully
if clean_data.empty:
    st.stop()

# Train model
model, accuracy, conf_matrix, class_report, X_test, y_test = train_model(clean_data)


# --- 2. SIDEBAR - USER INPUT ---
st.sidebar.header("🔬 Patient Information Input")
st.sidebar.markdown("Adjust the sliders to input patient data:")

def user_input_features():
    # Use min, max, and mean from the cleaned data for better slider defaults
    pregnancies = st.sidebar.slider('Pregnancies', 
                                    int(clean_data['Pregnancies'].min()), 
                                    int(clean_data['Pregnancies'].max()), 
                                    int(clean_data['Pregnancies'].mean()))
    
    glucose = st.sidebar.slider('Glucose (mg/dL)', 
                                int(clean_data['Glucose'].min()), 
                                int(clean_data['Glucose'].max()), 
                                int(clean_data['Glucose'].mean()))
    
    blood_pressure = st.sidebar.slider('Blood Pressure (mm Hg)', 
                                         int(clean_data['BloodPressure'].min()), 
                                         int(clean_data['BloodPressure'].max()), 
                                         int(clean_data['BloodPressure'].mean()))
    
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 
                                         int(clean_data['SkinThickness'].min()), 
                                         int(clean_data['SkinThickness'].max()), 
                                         int(clean_data['SkinThickness'].mean()))
    
    insulin = st.sidebar.slider('Insulin (μU/mL)', 
                                int(clean_data['Insulin'].min()), 
                                int(clean_data['Insulin'].max()), 
                                int(clean_data['Insulin'].mean()))
    
    bmi = st.sidebar.slider('BMI (Body Mass Index)', 
                            float(clean_data['BMI'].min()), 
                            float(clean_data['BMI'].max()), 
                            float(clean_data['BMI'].mean()), 
                            step=0.1)
    
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 
                            float(clean_data['DiabetesPedigreeFunction'].min()), 
                            float(clean_data['DiabetesPedigreeFunction'].max()), 
                            float(clean_data['DiabetesPedigreeFunction'].mean()), 
                            step=0.001, 
                            format="%.3f")
    
    age = st.sidebar.slider('Age (years)', 
                            int(clean_data['Age'].min()), 
                            int(clean_data['Age'].max()), 
                            int(clean_data['Age'].mean()))
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# --- 3. MAIN PANEL DISPLAY AND PREDICTION ---

# Ensure all columns are present before predicting (a safeguard)
input_df = input_df[clean_data.drop('Outcome', axis=1).columns]

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display Input
st.markdown("## 📋 Patient's Input Values")
st.dataframe(input_df.style.highlight_max(axis=0), use_container_width=True)

# Display Prediction
st.markdown("---")
st.markdown("## 🎯 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction[0] == 1:
        st.error("### ⚠️ DIABETIC")
        st.markdown("The model predicts that the patient **has diabetes** (Outcome = 1).")
    else:
        st.success("### ✅ NON-DIABETIC")
        st.markdown("The model predicts that the patient **does not have diabetes** (Outcome = 0).")

with col2:
    st.markdown("### 📊 Prediction Probability")
    st.metric("Probability of Diabetes", f"{prediction_proba[0][1]:.2%}")
    st.metric("Probability of No Diabetes", f"{prediction_proba[0][0]:.2%}")

# Display detailed probabilities
st.markdown("---")
st.markdown("## 📈 Detailed Probability Analysis")

prob_df = pd.DataFrame({
    'Class': ['Non-Diabetic (0)', 'Diabetic (1)'],
    'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
})

st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}), use_container_width=True)

# --- 4. MODEL PERFORMANCE ---
st.markdown("---")
st.markdown("## 🤖 Model Performance Metrics")

st.markdown(f"""
**Machine Learning Algorithm Used:** Random Forest Classifier
- **Number of Trees:** 100
- **Max Depth:** 10
- **Random State:** 42
""")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Model Accuracy (Test Set)", f"{accuracy:.2%}")

with col4:
    # Precision is the accuracy of positive predictions (Diabetic)
    st.metric("Precision (Diabetic)", f"{class_report['1']['precision']:.2%}")

with col5:
    # Recall is the fraction of all actual diabetics that were correctly identified
    st.metric("Recall (Diabetic)", f"{class_report['1']['recall']:.2%}")

# Confusion Matrix
st.markdown("### Confusion Matrix")
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    index=['Actual Non-Diabetic (0)', 'Actual Diabetic (1)'],
    columns=['Predicted Non-Diabetic (0)', 'Predicted Diabetic (1)']
)
st.dataframe(conf_matrix_df, use_container_width=True)

# Classification Report
st.markdown("### Detailed Classification Report")
report_df = pd.DataFrame(class_report).transpose()
st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Note:</strong> This prediction is based on the Pima Indians Diabetes Dataset and the Random Forest model's learned patterns. It should not be used as a substitute for professional medical advice.</p>
    <p><em>Developed with Streamlit</em></p>
</div>
""", unsafe_allow_html=True)