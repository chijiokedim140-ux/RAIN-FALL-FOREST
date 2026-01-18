import streamlit as st
import pickle
import numpy as np
import os

# ----------------------------
# 1️⃣ Page configuration
# ----------------------------
st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Rainfall Prediction System")
st.write("Predict rainfall amount based on weather parameters.")

# ----------------------------
# 2️⃣ Load the model
# ----------------------------
def load_model():
    # Absolute path to the model
    model_path = os.path.join(os.path.dirname(__file__), "rainfall_model.pkl")

    if not os.path.isfile(model_path):
        st.error("❌ rainfall_model.pkl NOT FOUND! Make sure it is in the same folder as app.py")
        st.stop()

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------------------------
# 3️⃣ Sidebar inputs
# ----------------------------
st.sidebar.header("Enter Weather Parameters")

temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 70.0)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
pressure = st.sidebar.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
cloud_cover = st.sidebar.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)

# ----------------------------
# 4️⃣ Predict rainfall
# ----------------------------
if st.button("Predict Rainfall"):
    input_data = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover]])
    rainfall = model.predict(input_data)[0]

    st.success(f"🌧️ Predicted Rainfall: {rainfall:.2f} mm")

    if rainfall > 50:
        st.warning("⚠️ Heavy rainfall expected. Possible flooding.")
    elif rainfall > 10:
        st.info("ℹ️ Moderate rainfall expected.")
    else:
        st.info("☀️ Low or no rainfall expected.")

# ----------------------------
# 5️⃣ Footer
# ----------------------------
st.markdown("---")
st.caption("Rainfall Prediction System using Streamlit & Random Forest")
