# F1 Race Weekend Predictor

A two-stage machine learning pipeline that predicts Formula 1 qualifying and race outcomes using practice session data.

**Live app:** [f1raceweekendpredictor.streamlit.app](https://f1raceweekendpredictor.streamlit.app/)

---

## How it works

1. **Stage 1 — Qualifying prediction:** An XGBoost model takes FP1/FP2/FP3 lap times, sector times, and championship standings to predict each driver's qualifying grid position.
2. **Stage 2 — Race prediction:** A LightGBM model uses the predicted (or actual) grid positions alongside practice pace data and form to predict finishing order.

Predictions are pre-computed after each race weekend and stored as JSON files so the app loads instantly.

---

## Running locally

### Requirements
- Python 3.10+

### Setup

```bash
git clone https://github.com/your-username/F1DataAnalyticsProject.git
cd F1DataAnalyticsProject

pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

---

## Retraining the models

If you want to retrain on updated data:

```bash
# 1. Rebuild the training datasets (downloads from FastF1)
python scripts/build_dataset.py --years {year ex. 2023}

# 2. Train qualifying model
python -m src.qualifying_model train

# 3. Train race model
python -m src.race_model train
```

---

## Generating predictions after a new race weekend

After a race weekend completes, run:

```bash
python scripts/generate_predictions.py --year 2026 --round 4
```

Then commit and push — the live app updates automatically.

```bash
git add data/predictions/2026_04.json
git commit -m "Add 2026 R4 predictions"
git push
```

---

## Project structure

```
app/                    Streamlit dashboard
src/
  data_loader.py        FastF1 + Ergast data fetching
  feature_engineering.py  Feature computation (qualifying + race)
  qualifying_model.py   Stage 1 XGBoost model
  race_model.py         Stage 2 LightGBM model
  pipeline.py           End-to-end prediction pipeline
scripts/
  build_dataset.py      Rebuild training CSVs
  generate_predictions.py  Pre-compute prediction JSON files
models/                 Trained model artefacts
data/predictions/       Pre-computed prediction JSON files (one per race)
cache/                  FastF1 session cache (gitignored)
```
