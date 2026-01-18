import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(
    page_title="Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered"
)

# 🚨 NO CACHE — FORCE FRESH LOAD
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "rainfall_model.pkl")

    st.write("🔍 Looking for model at:")
    st.code(model_path)

    if not os.path.isfile(model_path):
        st.error("❌ rainfall_model.pkl NOT FOUND")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()
