from flask import Flask, render_template, request
import pandas as pd
import pickle
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ckd_random_forest_model.sav"

# Load trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

NUMERIC_COLUMNS = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wc", "rc"
]

CATEGORICAL_COLUMNS = [
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet",
    "pe", "ane"
]

# Keep values consistent with what your model was trained on
CATEGORY_MAP = {
    "yes": "yes",
    "no": "no",
    "present": "present",
    "notpresent": "notpresent",
    "not present": "notpresent",
    "normal": "normal",
    "abnormal": "abnormal",
    "good": "good",
    "poor": "poor",
    "0": "no",
    "1": "yes"
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            patient = {key: request.form.get(key, "") for key in (
                "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba",
                "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
                "htn", "dm", "cad", "appet", "pe", "ane"
            )}

            input_df = pd.DataFrame([patient])

            # Convert numeric columns
            for col in NUMERIC_COLUMNS:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

            # Clean categorical values
            for col in CATEGORICAL_COLUMNS:
                input_df[col] = input_df[col].astype(str).str.strip().str.lower()
                input_df[col] = input_df[col].replace(CATEGORY_MAP)

            # Fill missing numeric values
            input_df[NUMERIC_COLUMNS] = input_df[NUMERIC_COLUMNS].fillna(
                input_df[NUMERIC_COLUMNS].median()
            )

            # If some category is unknown, keep it as a valid dummy value
            for col in CATEGORICAL_COLUMNS:
                input_df[col] = input_df[col].where(
                    input_df[col].isin(["yes", "no", "present", "notpresent",
                                        "normal", "abnormal", "good", "poor"]),
                    "no"
                )

            # Create dummy variables exactly like training pipeline
            input_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLUMNS, drop_first=True)

            # Match columns used during training
            input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

            # Predict
            result = int(model.predict(input_df)[0])
            prediction = "CKD" if result == 1 else "NOT CKD"

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)


if __name__ == "__main__":
    app.run(debug=True)