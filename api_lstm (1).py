import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import numpy
import sys
import pickle
import pandas as pd

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

# ---- Fix numpy pickle compatibility ----
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.numeric'] = numpy.core.numeric


# FLASK SETUP — threaded=True allows parallel requests

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# MODEL PATHS

HEALTH_H5       = os.path.join(BASE_DIR, "health_feature_forecast_lstm.h5")
EDU_H5          = os.path.join(BASE_DIR, "education_feature_forecast_lstm.h5")
GDP_GROWTH_H5   = os.path.join(BASE_DIR, "gdp_growth_delta_lstm.h5")
INFLATION_H5    = os.path.join(BASE_DIR, "inflation_delta_lstm.h5")
UNEMPLOYMENT_H5 = os.path.join(BASE_DIR, "unemployment_delta_lstm.h5")

ARIMA_PKL = os.path.join(BASE_DIR, "arima_population_model.pkl")

BASE_YEAR = 2025


# PATCH INPUTLAYER (Keras compatibility)

from keras.layers import InputLayer as KerasInputLayer

class PatchedInputLayer(KerasInputLayer):
    def __init__(self, *args, batch_shape=None, **kwargs):
        if batch_shape is not None and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = tuple(batch_shape)
        super().__init__(*args, **kwargs)


def build_custom_objects():
    custom = {"InputLayer": PatchedInputLayer}
    try:
        from keras.mixed_precision import Policy as _Policy
        custom["DTypePolicy"] = _Policy
        custom["Policy"]      = _Policy
    except Exception:
        pass
    return custom


# LOAD LSTM MODELS

def load_models():
    custom = build_custom_objects()
    with keras.utils.custom_object_scope(custom):
        health     = load_model(HEALTH_H5,       compile=False)
        edu        = load_model(EDU_H5,           compile=False)
        gdp        = load_model(GDP_GROWTH_H5,    compile=False)
        inflation  = load_model(INFLATION_H5,     compile=False)
        unemploy   = load_model(UNEMPLOYMENT_H5,  compile=False)

    models = {
        "Health Forecast - LSTM":     health,
        "Education Forecast - LSTM":  edu,
        "GDP Growth Forecast - LSTM": gdp,
        "Inflation - LSTM":           inflation,
        "Unemployment - LSTM":        unemploy,
    }
    print("LSTM models loaded:", list(models.keys()))
    return models


# LOAD ARIMAX POPULATION MODEL

def load_arima_model():
    with open(ARIMA_PKL, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "models"       in data: models = data["models"]
        elif "country_models" in data: models = data["country_models"]
        elif "arima_models"   in data: models = data["arima_models"]
        else:                          models = data
    else:
        raise ValueError("Invalid ARIMA model format")

    print("Population ARIMAX models loaded:", list(models.keys()))
    return models


MODELS       = load_models()
ARIMA_MODELS = load_arima_model()

# Lock for thread-safe Keras inference
import threading
KERAS_LOCK = threading.Lock()


# LSTM INPUT SHAPES

MODEL_SHAPES = {
    "Health Forecast - LSTM":     (1, 5, 24),
    "Education Forecast - LSTM":  (1, 3, 18),
    "GDP Growth Forecast - LSTM": (1, 5,  8),
    "Inflation - LSTM":           (1, 5,  8),
    "Unemployment - LSTM":        (1, 5, 10),
}


# POPULATION FORECAST FUNCTION

def predict_population(country, year, gdp):
    country = country.upper().strip()

    if country not in ARIMA_MODELS:
        raise ValueError(f"Country '{country}' not found")
    if year < BASE_YEAR:
        raise ValueError(f"Year must be >= {BASE_YEAR}")

    model      = ARIMA_MODELS[country]
    path_years = list(range(BASE_YEAR, year + 1))

    exog = pd.DataFrame({
        "Year":              path_years,
        "D_Expenditure_GDP": [gdp] * len(path_years),
    })

    pred_scaled = model.predict(n_periods=len(exog), exogenous=exog.values)

    # pmdarima returns a pandas Series — use iloc[-1] not [-1]
    if hasattr(pred_scaled, "iloc"):
        last_val = pred_scaled.iloc[-1]
    else:
        last_val = pred_scaled[-1]

    if last_val is None:
        raise ValueError("ARIMA model returned None")

    return float(last_val * 1_000_000)


# API ROUTES

@app.route("/models", methods=["GET"])
def models():
    return jsonify({
        "models": list(MODELS.keys()) + ["Population Forecast - ARIMAX"]
    })


@app.route("/population/countries", methods=["GET"])
def population_countries():
    return jsonify({"countries": list(ARIMA_MODELS.keys())})


@app.route("/schema", methods=["POST"])
def schema():
    data       = request.get_json(force=True)
    model_name = data.get("model")

    if model_name == "Population Forecast - ARIMAX":
        return jsonify({"inputs": ["country", "year", "gdp"],
                        "countries": list(ARIMA_MODELS.keys())})
    if model_name in MODELS:
        return jsonify({"inputs": ["expenditure"]})

    return jsonify({"error": "Unknown model"}), 400


# PREDICT ENDPOINT

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        print("\n=== REQUEST ===")
        print(data)
        print("===============\n")

        model_name = data.get("model")
        if not model_name:
            return jsonify({"error": "Missing field: model"}), 400
        model_name = model_name.strip()

        # ---------- POPULATION ARIMAX ----------
        if model_name == "Population Forecast - ARIMAX":
            country = data.get("country")
            year    = data.get("year")
            gdp     = data.get("gdp") or data.get("expenditure")

            if country is None: return jsonify({"error": "Missing field: country"}), 400
            if year    is None: return jsonify({"error": "Missing field: year"}),    400
            if gdp     is None: return jsonify({"error": "Missing field: gdp"}),     400

            year = int(year)
            try:
                gdp = float(gdp)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid gdp value"}), 400

            pop = predict_population(country, year, gdp)
            return jsonify({"country": country, "year": year, "gdp": gdp, "prediction": pop})

        # ---------- LSTM MODELS ----------
        if model_name not in MODELS:
            return jsonify({
                "error": "Invalid model",
                "available_models": list(MODELS.keys()) + ["Population Forecast - ARIMAX"]
            }), 400

        expenditure = data.get("expenditure")
        if expenditure is None:
            return jsonify({"error": "Missing field: expenditure"}), 400
        try:
            expenditure = float(expenditure)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid expenditure value"}), 400

        model = MODELS[model_name]
        shape = MODEL_SHAPES[model_name]

        x = np.zeros(shape, dtype=np.float32)
        x[0, :, 0] = expenditure

        # Keras inference is not thread-safe — serialize with a lock
        with KERAS_LOCK:
            y = model.predict(x, verbose=0)

        prediction = float(np.ravel(y)[0])
        return jsonify({"prediction": prediction})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# START SERVER  — threaded=True is the key change (used for the graph)

if __name__ == "__main__":
    print("Starting API server (threaded)...")
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        threaded=True,      # allows parallel requests from Streamlit
    )