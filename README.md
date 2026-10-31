# Crop Yield Prediction – CI/CD

A machine-learning project that predicts crop yield (hg/ha) from country/area, crop type, year, rainfall, pesticide usage and average temperature. GitHub Actions runs an ML quality gate, and the Flask API can be deployed on Render.

## Dataset
`yield_df.csv` contains the crop-yield dataset. The target column is `hg/ha_yield`. The project removes the exported index column `Unnamed: 0` and uses:
- Categorical features: `Area`, `Item`
- Numeric features: `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python quality_gate.py
python app.py
```

Open `http://127.0.0.1:5000/`.

## Prediction API

Example:

`/predict?area=India&item=Rice,%20paddy&year=2010&rain=1083&pesticides=17000&temp=25`

Parameters: `area`, `item`, `year`, `rain`, `pesticides`, `temp`.

The response contains `predicted_yield_hg_per_ha`. The prediction is an estimate from the trained model, not an agricultural guarantee.

## Models and quality gate

The training script compares six regressors: Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Decision Tree and K-Nearest Neighbors. It selects the highest test-set R² model and saves it as `model.joblib`; evaluation results are saved to `metrics.json`.

The GitHub Actions workflow runs `quality_gate.py`. The gate passes when the best model's held-out test R² is at least 0.90. To demonstrate a failing run, temporarily change `MIN_R2 = 0.90` in `quality_gate.py` to `MIN_R2 = 0.999`, commit, and observe the failed workflow. Restore `0.90` and commit again.

## Deploy on Render

Create a **Web Service** connected to this GitHub repository:
- Runtime: Python 3
- Branch: `main`
- Build Command: `pip install -r requirements.txt && python train_model.py`
- Start Command: `gunicorn app:app`
- Instance: Free
- Environment variable: `PYTHON_VERSION` = `3.11.9`

After deployment, open the service URL. The root route should return JSON with `"status": "running"`. Test the `/predict` example above.

> Note: Render's free service may spin down when idle. The first request after inactivity can take longer.

## CI/CD workflow

GitHub Actions runs on pushes and pull requests targeting `main`, installs dependencies, and executes the ML quality gate. Configure Render auto-deploy to wait for CI checks if that option is available for your service and plan.
