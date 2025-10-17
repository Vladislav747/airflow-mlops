import os
import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

os.getenv("MLFLOW_TRACKING_URI")

# Получим датасет California housing
housing = datasets.fetch_california_housing(as_frame=True)
# Объединим фичи и таргет в один np.array
data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

name = "MedHouseExp_2"
experiment_id = mlflow.create_experiment(name)
mlflow.set_experiment(experiment_id)


models = dict(zip(["RandomForest", "LinearRegression", "HistGB"], 
                  [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

# Сделать препроцессинг
# Разделить на фичи и таргет
X, y = data[FEATURES], data[TARGET]

# Разделить данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучить стандартизатор на train
scaler = StandardScaler()
X_train_fitted = scaler.fit_transform(X_train)
X_test_fitted = scaler.transform(X_test)

# Обучить стандартизатор на train
scaler = StandardScaler()
X_train_fitted = scaler.fit_transform(X_train)
X_test_fitted = scaler.transform(X_test)

def train_model(model, model_name, X_train, X_test, y_train, y_test):
    # Обучить модель
    model.fit(X_train, y_train)

    # Сделать predict
    prediction = model.predict(X_test)
    
    # Получить описание данных
    signature = infer_signature(X_test, prediction)

    # Сохранить модель в артифактори
    model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)

    # Сохранить метрики модели
    mlflow.evaluate(
        model_info.model_uri, 
        data=X_test, 
        targets=y_test.values, 
        model_type="regressor", 
        evaluators=["default"]
    )

 with mlflow.start_run(run_name="Parent_run", experiment_id=experiment_id, description="Parent run") as parent_run:
    for model in models.keys():
        with mlflow.start_run(run_name=model, experiment_id=experiment_id, nested=True) as child_run:
            train_model(models[model], model, X_train, X_test, y_train, y_test)
