import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import re # Import the regex module for validation

warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION AND DATA LOADING ---
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥", layout="wide")

# Title
st.title("🏥 Diabetes Prediction Application")
st.markdown("### Pima Indians Diabetes Dataset - Machine Learning Predictor")
st.markdown("---")

# Function to load data directly from the CSV file
@st.cache_data
def load_data():
    try:
        # Assuming the CSV file is in the same directory as this script
        df = pd.read_csv('diabetes.csv')
        # Standardize column names
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
    columns_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in columns_to_clean:
        df_clean[col] = df_clean[col].replace(0, np.nan) 
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


# --- 2. SIDEBAR - USER INPUT (MODIFIED TO USE TEXT INPUTS) ---
st.sidebar.header("🔬 Patient Information Input")
st.sidebar.markdown("Enter patient data (numeric, up to 2 decimal places):")

def get_validated_input(label, default_value, key):
    # Set the regex pattern to allow numbers with optional decimal, up to 2 places.
    # Pattern: ^-? : optional negative sign
    #          \d* : zero or more digits (allows for .50)
    #          (\.\d{1,2})? : optional decimal point followed by 1 or 2 digits
    #          $ : end of string
    pattern = r'^-?\d*(\.\d{1,2})?$'
    
    # Use text_input in the sidebar
    user_input = st.sidebar.text_input(label, value=str(round(default_value, 2)), key=key)
    
    if not user_input:
        st.sidebar.warning(f"Please enter a value for {label}.")
        return None
    
    # Check if the input string matches the desired pattern (numeric with max 2 decimals)
    if not re.fullmatch(pattern, user_input):
        st.sidebar.warning(f"Invalid format for **{label}**. Please enter a number (e.g., 100 or 0.75). Max 2 decimal places.")
        return None
        
    try:
        # Convert the validated string to a float
        return float(user_input)
    except ValueError:
        # This is a fallback, should be caught by regex, but good for safety
        st.sidebar.error(f"Critical error processing {label}.")
        return None

def user_input_features():
    # Use mean of cleaned data as default value
    defaults = clean_data.mean().to_dict()
    
    # Dictionary to hold the raw input values
    raw_inputs = {}
    
    # Get validated input for each feature
    raw_inputs['Pregnancies'] = get_validated_input('Pregnancies', defaults['Pregnancies'], 'p')
    raw_inputs['Glucose'] = get_validated_input('Glucose (mg/dL)', defaults['Glucose'], 'g')
    raw_inputs['BloodPressure'] = get_validated_input('Blood Pressure (mm Hg)', defaults['BloodPressure'], 'bp')
    raw_inputs['SkinThickness'] = get_validated_input('Skin Thickness (mm)', defaults['SkinThickness'], 'st')
    raw_inputs['Insulin'] = get_validated_input('Insulin (μU/mL)', defaults['Insulin'], 'ins')
    raw_inputs['BMI'] = get_validated_input('BMI (Body Mass Index)', defaults['BMI'], 'bmi')
    raw_inputs['DiabetesPedigreeFunction'] = get_validated_input('Diabetes Pedigree Function', defaults['DiabetesPedigreeFunction'], 'dpf')
    raw_inputs['Age'] = get_validated_input('Age (years)', defaults['Age'], 'age')
    
    # Check if all inputs are valid (not None)
    if all(v is not None for v in raw_inputs.values()):
        # Convert validated inputs into a DataFrame
        features = pd.DataFrame(raw_inputs, index=[0])
        return features
    else:
        # If any input is invalid/missing, return an empty DataFrame to stop prediction
        return pd.DataFrame()

# Get user input
input_df = user_input_features()

# Stop the app if input data is invalid (i.e., validation failed in the sidebar)
if input_df.empty:
    st.stop()


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
    <p><strong>Note:</strong> This prediction is based on the Diabetes Dataset and the Random Forest model's learned patterns. It should not be used as a substitute for professional medical advice.</p>
    <p><em>Developed by Kovii</em></p>
</div>
""", unsafe_allow_html=True)