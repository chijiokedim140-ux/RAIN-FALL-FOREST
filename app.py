import streamlit as st
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("rainfall_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("🌧️ Rainfall Prediction System")
st.write("Predict the amount of rainfall based on weather conditions.")

# Sidebar inputs
st.sidebar.header("Enter Weather Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
pressure = st.sidebar.number_input("Atmospheric Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
cloud_cover = st.sidebar.number_input("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0)

# Prediction button
if st.button("🔍 Predict Rainfall"):
    input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])
    
    rainfall = model.predict(input_data)[0]

    st.success(f"🌧️ Predicted Rainfall: **{rainfall:.2f} mm**")

    if rainfall > 50:
        st.warning("⚠️ Heavy rainfall expected. Possible flooding.")
    elif rainfall > 10:
        st.info("ℹ️ Moderate rainfall expected.")
    else:
        st.info("☀️ Low or no rainfall expected.")

# Footer
st.markdown("---")
st.caption("Rainfall Prediction System using Machine Learning & Streamlit")
