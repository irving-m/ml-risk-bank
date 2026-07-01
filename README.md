# Credit Risk Regression — Scotiabank Case Study

End-to-end regression pipeline to predict a numeric credit risk target (`TARGET`) from customer and loan features. The workflow compares interpretable linear models (GLM) against machine learning approaches, selects a winner via cross-validation, and evaluates it on a held-out test set with SHAP-based interpretation.

**Dataset:** 50,001 records with 12 numeric features (`X1`–`X12`), a district variable (`X9`), and a pre-defined train/test flag (`BASE`).

---

## Pipeline Overview

| Step | Notebook | What was done |
|------|----------|---------------|
| 1 | `01_eda.ipynb` | Exploratory data analysis |
| 2 | `02_preprocessing.ipynb` | Cleaning, feature engineering, train/test splits |
| 3 | `03_modeling_glm.ipynb` | Ridge & Lasso baseline models |
| 4 | `04_modeling_ml.ipynb` | Random Forest, XGBoost, LightGBM, MLP |
| 5 | `05_evaluation.ipynb` | Final model test evaluation & SHAP |

All modeling steps log experiments to **MLflow** (`scotiabank_ml` experiment).

---

## Step 1 — Exploratory Data Analysis

**Notebook:** `notebooks/01_eda.ipynb`

- Loaded `data/raw/data_modelo.csv` (50,001 rows × 15 columns).
- Profiled missing values and **sentinel codes** used in place of nulls:
  - `X3`, `X4`: `-9999998` (~40% and ~30% of rows respectively)
  - `X11`: `-99000720`, `-99000632`
  - `X12`: `-99000792`
  - `X8`: `999`
- Mapped overlap between `X3` and `X4` missingness (10,397 rows missing both).
- Plotted distributions, histograms, and boxplots for `TARGET` and all numeric features.
- Quantified outliers via IQR fences; identified heavy right tails on `X1`–`X5`.
- Ran correlation analysis — found **X5 highly correlated with X4** (ρ ≈ 0.96), leading to its removal in preprocessing.
- Explored geographic signal: 630 districts in `X9`, with meaningful variation in median `TARGET`.
- Verified train vs. test `TARGET` distribution alignment using the `BASE` column.

---

## Step 2 — Preprocessing & Feature Engineering

**Notebook:** `notebooks/02_preprocessing.ipynb`

| Action | Detail |
|--------|--------|
| Sentinel handling | Replaced sentinel values with `NaN`; created `X3_missing` and `X4_missing` indicator flags |
| Feature drop | Removed `X5` (multicollinearity with `X4`) |
| Outlier capping | Clipped `X1`–`X4` at the 99th percentile |
| Log transform | Applied `log1p` to `TARGET`, `X1`, `X2`, `X3`, `X4`, `X10`, `X11` |
| Imputation | Median imputation on numeric columns, **fit on train only** |
| Categorical encoding | Smoothed target encoding on district (`X9`, k=10), **fit on train only** |
| Scaling | `StandardScaler` on continuous features, **fit on train only** |

**Outputs:**
- `data/processed/data_processed.csv` — full cleaned dataset
- `data/splits/train.csv` — 33,334 rows
- `data/splits/test.csv` — 16,667 rows

**Feature set (13 features):** `X1`, `X2`, `X3`, `X4`, `X6`, `X7`, `X8`, `X10`, `X11`, `X12`, `X3_missing`, `X4_missing`, `X9_encoded`

---

## Step 3 — GLM Modeling

**Notebook:** `notebooks/03_modeling_glm.ipynb`

Trained two regularized linear models with 5-fold cross-validation. Metrics are on the **log-transformed target**.

| Model | Val RMSE | Val R² |
|-------|----------|--------|
| Ridge (α=1.0) | 0.5454 ± 0.0042 | 0.3083 ± 0.0097 |
| Lasso (α=0.001) | 0.5454 ± 0.0042 | 0.3083 ± 0.0094 |

Both models show similar performance with no overfitting (train ≈ validation). They serve as the interpretable baseline for comparison with ML models.

---

## Step 4 — Machine Learning Modeling

**Notebook:** `notebooks/04_modeling_ml.ipynb`

Compared four non-linear models using the same 13-feature set and 5-fold CV:

| Model | Val RMSE | Val R² |
|-------|----------|--------|
| **XGBoost** | **0.5066 ± 0.0055** | **0.4033 ± 0.0140** |
| LightGBM | 0.5105 ± 0.0054 | 0.3941 ± 0.0130 |
| Random Forest | 0.5151 ± 0.0042 | 0.3830 ± 0.0104 |
| MLP (128-64-32) | 0.5234 ± 0.0029 | 0.3630 ± 0.0105 |

**XGBoost** achieved the best cross-validation performance (~18% RMSE improvement over the GLM baseline) and was selected as the final model.

The winning model is serialized to `models/ml/final_model_xgb.joblib`. Reusable training and evaluation scripts live in `src/train.py` and `src/evaluate.py`.

---

## Step 5 — Final Evaluation & Interpretation

**Notebook:** `notebooks/05_evaluation.ipynb`

Evaluated the XGBoost model on the held-out test set:

| Metric | Log scale | Original scale |
|--------|-----------|----------------|
| RMSE | 0.5008 | 4,808 |
| MAE | 0.3913 | 3,122 |
| R² | 0.3953 | — |

Additional analysis:
- **Predicted vs. actual** scatter plot on original-scale `TARGET`
- **Residual diagnostics** — distribution and residuals vs. actual value
- **Error by segment** — MAE increases in higher `TARGET` quartiles (Q4 MAE ≈ 6,472 vs. Q1 ≈ 2,074)
- **SHAP values** — global feature importance and directional effects via TreeExplainer

---

## Project Structure

```
ml_risk_scotiabank/
├── data/
│   ├── raw/                        # Original data (data_modelo.csv)
│   ├── processed/                  # Cleaned dataset (data_processed.csv)
│   └── splits/                     # Train/test CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling_glm.ipynb
│   ├── 04_modeling_ml.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── train.py                    # Train & save final XGBoost model
│   └── evaluate.py                 # Load model and log test metrics
├── models/
│   ├── glm/                        # GLM artifacts (via MLflow)
│   └── ml/
│       └── final_model_xgb.joblib  # Serialized winning model
├── mlartifacts/                    # MLflow model artifacts
├── mlflow.db                       # Local MLflow tracking store
└── requirements.txt
```

---

## How to Run

1. **Set up the environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

2. **Place raw data** — Ensure `data_modelo.csv` is in `data/raw/`.

3. **Run notebooks in order** — Execute `01_eda.ipynb` through `05_evaluation.ipynb`.

4. **Optional: MLflow UI** — Start the tracking server to browse experiments:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Then open `http://127.0.0.1:5000`.

5. **Optional: retrain & evaluate via scripts**
   ```bash
   cd src
   python train.py      # trains XGBoost on full train set, saves to models/ml/
   python evaluate.py   # loads model, evaluates on test set, logs to MLflow
   ```

---

## Key Takeaways

- Sentinel values in `X3`/`X4` were a major data quality issue (~40% / ~30% of rows); explicit missingness flags preserved this signal for modeling.
- Tree-based models outperformed linear baselines, with XGBoost delivering the best balance of accuracy and stability across CV folds.
- Test R² of ~0.40 indicates moderate predictive power; error is concentrated in high-`TARGET` segments, suggesting room for segment-specific modeling or additional features.
- SHAP analysis provides actionable feature-level explanations for the selected XGBoost model.
