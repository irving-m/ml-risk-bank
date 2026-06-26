import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("scotiabank_ml")

train = pd.read_csv("../data/splits/train.csv")
model_save_path = "../models/ml/final_model_xgb.joblib"
FEATURES = ["X1","X2","X3","X4","X6","X7","X8","X10","X11","X12",
            "X3_missing","X4_missing","X9_encoded"]
TARGET = "TARGET"

X_train, y_train = train[FEATURES], train[TARGET]

final_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

with mlflow.start_run(run_name="xgboost_final"):
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("trained_on", "full_train_100pct")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 4)
    mlflow.log_param("subsample", 0.8)
    mlflow.log_param("colsample_bytree", 0.8)
    mlflow.sklearn.log_model(final_model, "final_model")

joblib.dump(final_model, model_save_path)
print("Model saved to final_model_xgb.joblib")

