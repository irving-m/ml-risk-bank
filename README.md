# Credit Risk Regression — Scotiabank Case Study

## Description

This project trains and compares two regression models—a Generalized Linear Model (GLM) and a machine learning model—to predict a numeric credit risk target variable. The workflow follows a structured pipeline from data ingestion through model evaluation, ending with a PowerPoint presentation as the final deliverable.

## Folder Structure

```
ml_risk_scotiabank/
├── data/
│   ├── raw/                        # Original unmodified data (data_modelo.csv)
│   ├── processed/                  # Cleaned and feature-engineered datasets
│   └── splits/                     # Train/validation split files
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_preprocessing.ipynb      # Missing values, outliers, transformations
│   ├── 03_modeling_glm.ipynb       # GLM model training and evaluation
│   ├── 04_modeling_ml.ipynb        # ML model training and evaluation
│   └── 05_comparison_interpretation.ipynb  # Model comparison and business interpretation
├── src/
│   ├── preprocessing.py            # Reusable preprocessing functions
│   ├── features.py                 # Feature engineering logic
│   ├── train.py                    # Model training routines
│   ├── evaluate.py                 # Metrics: RMSE, MAE, R², KS, PSI, etc.
│   └── utils.py                    # General helpers (logging, I/O, config loading)
├── models/
│   ├── glm/                        # Serialized GLM model artifacts
│   └── ml/                         # Serialized ML model artifacts
├── outputs/
│   ├── figures/                    # Plots and charts from analysis
│   ├── metrics/                    # Model performance results (CSV/JSON)
│   └── presentation/               # Final PowerPoint deliverable
└── config/
    └── config.yaml                 # Paths, hyperparameters, thresholds
```

## How to Run

1. **Set up the environment** — Create a virtual environment and install dependencies from `requirements.txt`.
2. **Add raw data** — Place `data_modelo.csv` in `data/raw/`.
3. **Configure the project** — Review and update `config/config.yaml` (paths, model parameters, preprocessing thresholds).
4. **Run the notebooks in order** — Execute `01_eda.ipynb` through `05_comparison_interpretation.ipynb`.
5. **Review outputs** — Check `outputs/figures/` and `outputs/metrics/` for artifacts; the final presentation is saved to `outputs/presentation/`.
