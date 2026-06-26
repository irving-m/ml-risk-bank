import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("scotiabank_ml")

test = pd.read_csv("../data/splits/test.csv")

FEATURES = ["X1","X2","X3","X4","X6","X7","X8","X10","X11","X12",
            "X3_missing","X4_missing","X9_encoded"]
TARGET = "TARGET"

X_test, y_test = test[FEATURES], test[TARGET]

final_model = joblib.load("../models/ml/final_model_xgb.joblib")

y_pred = final_model.predict(X_test)

# Log scale metrics
test_rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# Original scale metrics
test_rmse_ogscale = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))

with mlflow.start_run(run_name="xgboost_final_test_eval"):
    mlflow.log_metric("test_rmse_log", test_rmse_log)
    mlflow.log_metric("test_r2", test_r2)
    mlflow.log_metric("test_rmse_ogscale", test_rmse_ogscale)

print(f"Test | RMSE (log): {test_rmse_log:.4f} | R²: {test_r2:.4f} | RMSE - original scale: {test_rmse_ogscale:.0f}")

