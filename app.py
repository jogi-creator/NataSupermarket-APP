import streamlit as st
import pandas as pd
import joblib, json
st.set_page_config(page_title="NATA Supermarket - Customer Spend Predictor", layout="centered")
st.title("NATA Supermarket — Customer Spend Predictor")
MODEL_PATH = "model.pkl"
FEATURES_PATH = "feature_columns.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)

model = load_model()
feature_info = load_features()

st.sidebar.header("Input customer info")
input_data = {}

for col in feature_info.get("numeric", []):
    input_data[col] = st.sidebar.number_input(col, value=0.0, format="%.2f")

for col, options in feature_info.get("categorical", {}).items():
    choice = st.sidebar.selectbox(col, options)
    input_data[col] = choice

if st.sidebar.button("Predict spending"):
    df_input = pd.DataFrame([input_data])
    for n in feature_info.get("numeric", []):
        try:
            df_input[n] = pd.to_numeric(df_input[n])
        except Exception:
            pass
    pred = model.predict(df_input)[0]
    st.subheader("Predicted total spend (last 2 years)")
    st.write("₹{:, .2f}".format(pred))
    st.caption("Model: Gradient Boosting regressor trained on historical data")

st.markdown("---")
st.markdown("**Instructions:** Use the sidebar to set customer values, then click Predict. The model file (model.pkl) and feature metadata (feature_columns.json) must be in the same folder as this app.")

