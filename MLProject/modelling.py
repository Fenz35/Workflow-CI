import mlflow
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Create a new MLflow Experiment
mlflow.set_experiment("Gaming_Academic_Performance_Basic")

# Memuat Data yang sudah diproses
data_path = "train_cleaned.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data yang sudah diproses tidak ditemukan di {data_path}")

data = pd.read_csv(data_path)

X = data.drop("grades", axis=1)
y = data["grades"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)

input_example = X_train.iloc[:5]

with mlflow.start_run(run_name="GBR_Default_Run"):
    mlflow.autolog(log_models=False)
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
       
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    
    # kalkulasi metrik pada data testing
    predictions = model.predict(X_test)
    
    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Explicitly log additional metrics to tracking UI
    mlflow.log_metric("testing_r2_score", r2)
    mlflow.log_metric("testing_mae", mae)
    mlflow.log_metric("testing_rmse", rmse)
    
    print("--- Training Metrics ---")
    print(f"R2 Score : {r2:.4f}")
    print(f"MAE      : {mae:.4f} (Average off by {mae:.2f} grade points)")
    print(f"RMSE     : {rmse:.4f}")