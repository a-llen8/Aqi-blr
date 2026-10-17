# Bengaluru AQI Prediction

ML-based Air Quality Index prediction for Bengaluru (6 CPCB stations, 2023-07-01 to 2026-06-30).

## Pipeline
Run the notebooks in order, with `notebooks/` as the working folder:

1. `01_preprocessing.ipynb` - raw CPCB files (`data/raw`) to a combined, gap-filled dataset
2. `02_eda_weather_merge.ipynb` - data repair, weather merge, EDA (outputs in `eda_outputs/`)
3. `03_aqi_features_models.ipynb` - CPCB AQI, feature engineering, forecasting models

## Folders
- `data/raw` - source station exports
- `data/processed` - pipeline outputs, incl. `feature_dictionary.csv`
- `eda_outputs` - EDA tables and plots
- `docs/project_notes.md` - methodology notes and known data-quality issues
