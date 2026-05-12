import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import os

# Create a new MLflow Experiment
mlflow.set_experiment("Gaming_Academic_Performance_CI")

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

with mlflow.start_run(run_name="GBR_CI"):
    mlflow.autolog()
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
       
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    
    # Kalkulasi metric R2 Score dan log ke MLflow
    r2 = model.score(X_test, y_test)
    print(f"Model trained with R2 Score: {r2:.4f}")