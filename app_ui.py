import streamlit as st
import requests
import os 
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Disease Prediction System")
st.markdown("Fill in your health details below")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age    = st.number_input("Age",         min_value=1,  max_value=120, value=30)
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
        gender = st.selectbox("Gender",         ["Male", "Female"])
        blood_pressure = st.selectbox("Blood Pressure", ["Low", "Normal", "High"])

    with col2:
        cholesterol         = st.selectbox("Cholesterol",         ["Low", "Normal", "High"])
        glucose             = st.selectbox("Glucose",             ["Low", "Normal", "High"])
        smoking             = st.selectbox("Smoking",             ["Yes", "No"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
        exercise            = st.selectbox("Exercise",            ["Yes", "No"])
        family_history      = st.selectbox("Family History",      ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict Diseases", use_container_width=True)

if submitted:
    payload = {
        "age":                age,
        "gender":             gender,
        "height":             height,
        "weight":             weight,
        "blood_pressure":     blood_pressure,
        "cholesterol":        cholesterol,
        "glucose":            glucose,
        "smoking":            smoking,
        "alcohol_consumption":alcohol_consumption,
        "exercise":           exercise,
        "family_history":     family_history
    }

    with st.spinner("Analyzing..."):
        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            predictions = response.json()["predictions"]

            st.markdown("---")
            st.subheader("📊 Results")

            col1, col2 = st.columns(2)
            diseases   = list(predictions.items())
            half       = len(diseases) // 2

            for i, (disease, result) in enumerate(diseases):
                col  = col1 if i < half else col2
                name = disease.replace("_", " ").title()
                with col:
                    if result == 1:
                        st.error(f"🔴 {name}: Detected")
                    else:
                        st.success(f"🟢 {name}: Not Detected")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure FastAPI is running!")
