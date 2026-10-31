"""Train and compare six crop-yield regression models."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "yield_df.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
TARGET = "hg/ha_yield"
CATEGORICAL = ["Area", "Item"]
NUMERIC = ["Year", "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
FEATURES = CATEGORICAL + NUMERIC


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data[required].dropna()
    return data[FEATURES], data[TARGET]


def make_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])


def train_and_evaluate():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42, min_samples_leaf=2
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    results = []
    best_model, best_r2, best_name = None, float("-inf"), None
    for name, estimator in models.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        r2 = float(r2_score(y_test, predictions))
        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
        results.append({
            "model": name, "r2": r2, "mae": mae, "rmse": rmse
        })
        print(f"{name}: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
        if r2 > best_r2:
            best_model, best_r2, best_name = pipeline, r2, name

    joblib.dump(best_model, MODEL_PATH)
    report = {
        "best_model": best_name,
        "best_r2": best_r2,
        "minimum_required_r2": 0.90,
        "models": sorted(results, key=lambda item: item["r2"], reverse=True),
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nBest model: {best_name} (R2={best_r2:.4f})")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    return report


if __name__ == "__main__":
    train_and_evaluate()
