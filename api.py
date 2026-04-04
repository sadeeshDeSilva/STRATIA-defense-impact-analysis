from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import h5py
import os
import numpy as np

# Keras / TensorFlow for LSTM
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)


def load_rf_pickle_from_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "sklearn_pipeline_pickle" not in f:
            raise KeyError(
                f"Dataset 'sklearn_pipeline_pickle' not found in {h5_path}. "
                "Make sure you saved the pipeline bytes under that key."
            )
        model_bytes = bytes(f["sklearn_pipeline_pickle"][()])
    return pickle.loads(model_bytes)


def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    models = {}

    # ----- Random Forest (sklearn pipeline inside .h5) -----
    rf_path = os.path.join(base_dir, "RF_CO2_Final_fixed.h5")
    models["CO2 - Random Forest"] = load_rf_pickle_from_h5(rf_path)

    # ----- LSTM models (Keras .h5) -----
    health_path = os.path.join(base_dir, "health_feature_forecast_lstm.h5")
    edu_path = os.path.join(base_dir, "education_feature_forecast_lstm.h5")

    models["Health Forecast - LSTM"] = load_model(health_path, compile=False)
    models["Education Forecast - LSTM"] = load_model(edu_path, compile=False)

    print("✅ All models loaded successfully!")
    return models


MODELS = load_models()


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({"models": list(MODELS.keys())})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}

    # Basic validation
    required = ["model", "country", "year", "expenditure"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    model_name = data["model"]
    if model_name not in MODELS:
        return jsonify({"error": f"Invalid model '{model_name}'"}), 400

    country = str(data["country"]).strip().upper()
    year = int(data["year"])
    expenditure = float(data["expenditure"])

    model = MODELS[model_name]

    # ----- Random Forest pathway -----
    if model_name == "CO2 - Random Forest":
        input_df = pd.DataFrame({
            "Country": [country],
            "Year": [year],
            "D_Expenditure_GDP": [expenditure]
        })
        pred = float(model.predict(input_df)[0])
        return jsonify({"prediction": pred})

    # ----- LSTM pathway -----
    x = np.array([expenditure], dtype=np.float32).reshape((1, 1, 1))
    y = model.predict(x, verbose=0)

    pred = float(np.ravel(y)[0])
    return jsonify({"prediction": pred})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)