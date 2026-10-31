"""Flask API for crop-yield prediction."""
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from train_model import MODEL_PATH, train_and_evaluate

app = Flask(__name__)


def get_model():
    if not Path(MODEL_PATH).exists():
        train_and_evaluate()
    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Crop Yield Prediction API is live",
        "usage": "/predict?area=India&item=Rice,%20paddy&year=2010&rain=1083&pesticides=17000&temp=25",
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.get("/predict")
def predict():
    required = ["area", "item", "year", "rain", "pesticides", "temp"]
    missing = [key for key in required if request.args.get(key) is None]
    if missing:
        return jsonify({
            "error": "Missing required query parameters",
            "missing": missing,
            "required": required,
        }), 400
    try:
        row = pd.DataFrame([{
            "Area": request.args["area"],
            "Item": request.args["item"],
            "Year": int(request.args["year"]),
            "average_rain_fall_mm_per_year": float(request.args["rain"]),
            "pesticides_tonnes": float(request.args["pesticides"]),
            "avg_temp": float(request.args["temp"]),
        }])
        prediction = float(get_model().predict(row)[0])
        return jsonify({
            "status": "success",
            "area": request.args["area"],
            "crop": request.args["item"],
            "year": int(request.args["year"]),
            "predicted_yield_hg_per_ha": round(prediction, 2),
        })
    except ValueError:
        return jsonify({"error": "year, rain, pesticides and temp must be numeric"}), 400
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "details": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
