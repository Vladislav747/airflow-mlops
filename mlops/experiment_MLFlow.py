import os
import numpy as np
import pandas as pd

import mlflow
from mlflow.models import infer_signature

from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression, LinearRegression
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
