import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="CardioApp | Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model and scaler
@st.cache_resource
def load_model():
    try:
        data = joblib.load("best_heart_disease_model.pkl")
        return data['model'], data['scaler']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffcccc;
        border: 2px solid #ff4b4b;
        color: #b30000;
    }
    .low-risk {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">❤️ CardioApp Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter your health metrics below to assess your cardiovascular disease risk using our Random Forest ML Model.</p>', unsafe_allow_html=True)

    # Create two columns for the layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👤 Personal Details")
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        
        st.subheader("🩺 Clinical Metrics")
        cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], help="1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "True (> 120 mg/dl)" if x == 1 else "False (< 120 mg/dl)")
        
    with col2:
        st.subheader("📈 Diagnostic Tests")
        restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2], help="0: normal, 1: ST-T wave abnormality, 2: probable left ventricular hypertrophy")
        thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[1, 2, 3], help="1: upsloping, 2: flat, 3: downsloping")
        ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7], help="3: normal, 6: fixed defect, 7: reversable defect")

    st.markdown("---")

    # Prediction Button
    if st.button("Predict Cardiovascular Risk"):
        if model is None or scaler is None:
            st.error("Model failed to load. Please ensure 'best_heart_disease_model.pkl' exists.")
            return

        # Prepare the input array
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Depending on the model, it might also have predict_proba
        try:
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = probabilities[prediction] * 100
        except:
            confidence = None

        # Display result
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card high-risk">
                    <h2>⚠️ High Risk Detected</h2>
                    <p>The model predicts that you have a high risk of cardiovascular disease.</p>
                    {'<p><strong>Confidence: {:.1f}%</strong></p>'.format(confidence) if confidence else ''}
                    <p><em>Please consult with a healthcare professional immediately.</em></p>
                </div>
            """, unsafe_allow_html=True)
            st.snow() # Just an effect, but maybe balloon is better for success. For High Risk, no effect is fine or error.
        else:
            st.markdown(f"""
                <div class="result-card low-risk">
                    <h2>✅ Low Risk Detected</h2>
                    <p>The model predicts that you have a low risk of cardiovascular disease.</p>
                    {'<p><strong>Confidence: {:.1f}%</strong></p>'.format(confidence) if confidence else ''}
                    <p><em>Maintain a healthy lifestyle!</em></p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

if __name__ == "__main__":
    main()
