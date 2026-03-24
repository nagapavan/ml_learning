import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Page Config ---
st.set_page_config(
    page_title="Insurance Charge Predictor", page_icon="🏥", layout="centered"
)

st.title("🏥 Medical Insurance Charge Predictor")
st.markdown("Enter your details below to get an estimated insurance charge.")
st.divider()


# --- Load the trained model ---
# You'll need to save your best model first (see instructions below)
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = f.read()
    return pickle.loads(model)


try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning(
        "⚠️ Model file `best_model.pkl` not found. See instructions below to save your model."
    )

# --- User Input Form ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input(
        "Number of Children", min_value=0, max_value=10, value=0, step=1
    )
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox(
        "Region", options=["northeast", "northwest", "southeast", "southwest"]
    )

st.divider()

# --- Predict ---
if st.button("💡 Predict Insurance Charge", type="primary", use_container_width=True):
    if model_loaded:
        # Build input dataframe matching training format
        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                }
            ]
        )

        # Apply the same encoding as training
        # Adjust this section to match YOUR preprocessing pipeline
        label_cols = ["sex", "smoker", "region"]
        for col in label_cols:
            le = LabelEncoder()
            # Fit on all known categories
            if col == "sex":
                le.fit(["female", "male"])
            elif col == "smoker":
                le.fit(["no", "yes"])
            elif col == "region":
                le.fit(["northeast", "northwest", "southeast", "southwest"])
            input_data[col] = le.transform(input_data[col])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Display result
        st.success(f"### 💰 Estimated Insurance Charge: **${prediction:,.2f}**")
    else:
        st.error("Please save your trained model first (see below).")


# # Use your best performing model (e.g., the tuned XGBoost)
# best_model = best_models["XGBRegressor"]
# with open("best_model.pkl", "wb") as f:
#     pickle.dump(best_model, f)
