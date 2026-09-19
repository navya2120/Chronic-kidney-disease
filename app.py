import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Nephro AI",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    try:
        with open("ckd_random_forest_model.sav", "rb") as file:
            return pickle.load(file)

    except Exception as e:
        st.error("Model loading failed")
        st.code(str(e))
        st.stop()
# -----------------------------
# Title
# -----------------------------
st.title("🩺 NEPHRO AI")
st.subheader("Chronic Kidney Disease Classification using Machine Learning")

st.write(
    "Enter the patient's medical information below to get "
    "the model's classification."
)

st.divider()

# -----------------------------
# Patient Information
# -----------------------------
st.header("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1.0, max_value=120.0, value=45.0)
    bp = st.number_input("Blood Pressure", value=80.0)
    sg = st.number_input("Specific Gravity", value=1.020, format="%.3f")
    al = st.number_input("Albumin", min_value=0.0, max_value=5.0, value=0.0)
    su = st.number_input("Sugar", min_value=0.0, max_value=5.0, value=0.0)

with col2:
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose", value=120.0)
    bu = st.number_input("Blood Urea", value=40.0)
    sc = st.number_input("Serum Creatinine", value=1.2)

with col3:
    sod = st.number_input("Sodium", value=140.0)
    pot = st.number_input("Potassium", value=4.5)
    hemo = st.number_input("Hemoglobin", value=13.0)
    pcv = st.number_input("Packed Cell Volume", value=40.0)
    wc = st.number_input("White Blood Cell Count", value=8000.0)
    rc = st.number_input("Red Blood Cell Count", value=5.0)

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    htn = st.selectbox("Hypertension", ["no", "yes"])

with col5:
    dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])

with col6:
    cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])

col7, col8 = st.columns(2)

with col7:
    appet = st.selectbox("Appetite", ["good", "poor"])

with col8:
    pe = st.selectbox("Pedal Edema", ["no", "yes"])

ane = st.selectbox("Anemia", ["no", "yes"])

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict CKD", use_container_width=True):

    try:

        # Create input dataframe
        input_data = {
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "su": su,
            "rbc": rbc,
            "pc": pc,
            "pcc": pcc,
            "ba": ba,
            "bgr": bgr,
            "bu": bu,
            "sc": sc,
            "sod": sod,
            "pot": pot,
            "hemo": hemo,
            "pcv": pcv,
            "wc": wc,
            "rc": rc,
            "htn": htn,
            "dm": dm,
            "cad": cad,
            "appet": appet,
            "pe": pe,
            "ane": ane
        }

        input_df = pd.DataFrame([input_data])

        # Same preprocessing used in notebook
        numeric_columns = [
            "age", "bp", "sg", "al", "su",
            "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc"
        ]

        for column in numeric_columns:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="coerce"
            )

        categorical_columns = [
            "rbc", "pc", "pcc", "ba",
            "htn", "dm", "cad", "appet",
            "pe", "ane"
        ]

        mapping = {
            "yes": 1,
            "no": 0,
            "present": 1,
            "notpresent": 0,
            "normal": 1,
            "abnormal": 0,
            "good": 1,
            "poor": 0
        }

        for column in categorical_columns:
            input_df[column] = input_df[column].map(mapping)

        # Convert categorical columns to dummy variables
        input_df = pd.get_dummies(
            input_df,
            drop_first=True
        )

        # Match model's training columns
        input_df = input_df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Prediction
        result = model.predict(input_df)[0]

        st.divider()

        if result == 1:

            st.error("⚠️ CKD")

            st.write(
                "The model classified this patient record as CKD."
            )

        else:

            st.success("✅ NOT CKD")

            st.write(
                "The model classified this patient record as NOT CKD."
            )

    except Exception as e:

        st.error("Prediction Error")
        st.code(str(e))

# -----------------------------
# Disclaimer
# -----------------------------
st.divider()

st.caption(
    "Note: This is an educational machine-learning project "
    "and should not be used as a medical diagnosis."
)