import mlflow
import pandas as pd
import numpy as np
import random
import os
import warnings
import sys

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Memuat Data yang sudah diproses
    file_path = "train_cleaned.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found at {file_path}")
    
    data = pd.read_csv(file_path)

    X = data.drop("grades", axis=1)
    y = data["grades"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
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
