import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import json

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #6a0dad;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-card {
    background-color: #f0f8ff;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #6a0dad;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model(model_format="joblib"):
    try:
        if model_format == "joblib":
            model = joblib.load("models/iris_model.joblib")
        else:
            with open("models/iris_model.pickle", "rb") as f:
                model = pickle.load(f)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# -----------------------------
# Load Model Information
# -----------------------------
@st.cache_data
def load_model_info():
    try:
        with open("models/model_info.json", "r") as f:
            return json.load(f)

    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None


# -----------------------------
# Load Feature Ranges
# -----------------------------
@st.cache_data
def load_feature_ranges():
    try:
        with open("models/feature_ranges.json", "r") as f:
            data = json.load(f)

        # Add default values if missing
        for feature in data:
            if "default" not in data[feature]:
                data[feature]["default"] = (
                    data[feature]["min"] +
                    data[feature]["max"]
                ) / 2

        return data

    except Exception:
        return {
            "sepal_length": {
                "min": 4.3,
                "max": 7.9,
                "default": 5.8
            },
            "sepal_width": {
                "min": 2.0,
                "max": 4.4,
                "default": 3.0
            },
            "petal_length": {
                "min": 1.0,
                "max": 6.9,
                "default": 4.0
            },
            "petal_width": {
                "min": 0.1,
                "max": 2.5,
                "default": 1.2
            }
        }


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_format = st.radio(
        "Select Model Format",
        ["joblib", "pickle"]
    )

# Load Files
model = load_model(model_format)
model_info = load_model_info()
feature_ranges = load_feature_ranges()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

st.write(
    """
    Predict the species of an Iris flower using a
    Random Forest machine learning model.
    """
)

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        float(feature_ranges["sepal_length"]["min"]),
        float(feature_ranges["sepal_length"]["max"]),
        float(feature_ranges["sepal_length"]["default"]),
        0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        float(feature_ranges["sepal_width"]["min"]),
        float(feature_ranges["sepal_width"]["max"]),
        float(feature_ranges["sepal_width"]["default"]),
        0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        float(feature_ranges["petal_length"]["min"]),
        float(feature_ranges["petal_length"]["max"]),
        float(feature_ranges["petal_length"]["default"]),
        0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        float(feature_ranges["petal_width"]["min"]),
        float(feature_ranges["petal_width"]["max"]),
        float(feature_ranges["petal_width"]["default"]),
        0.1
    )

with col2:
    st.subheader("Current Values")

    values_df = pd.DataFrame({
        "Feature": [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width"
        ],
        "Value": [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(values_df, hide_index=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🎯 Predict Species"):

    if model is not None and model_info is not None:

        input_data = pd.DataFrame(
            [[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]],
            columns=model_info["feature_names"]
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        species = model_info["target_names"][prediction]

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.success(f"Predicted Species: {species}")

        st.subheader("Confidence Scores")

        for i, prob in enumerate(probabilities):
            class_name = model_info["target_names"][i]
            st.write(f"{class_name}: {prob:.2%}")
            st.progress(float(prob))

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    else:
        st.error("Model files could not be loaded.")

# -----------------------------
# About Dataset
# -----------------------------
with st.expander("📚 About Iris Dataset"):
    st.write("""
    Iris is a classic machine learning dataset introduced by
    Ronald Fisher in 1936.

    Features:
    - Sepal Length
    - Sepal Width
    - Petal Length
    - Petal Width

    Classes:
    - Setosa
    - Versicolor
    - Virginica
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center>Built with Streamlit and Scikit-Learn</center>",
    unsafe_allow_html=True
)