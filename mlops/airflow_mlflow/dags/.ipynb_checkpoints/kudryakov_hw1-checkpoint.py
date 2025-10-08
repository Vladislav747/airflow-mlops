from datetime import datetime
import io
import logging
import os
import tempfile
import requests
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "s3_connection"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = "VLADISLAV"
MY_SURNAME = "KUDRYAKOV"

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"

logging.basicConfig(filename="first_dag.log", level=logging.INFO)
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


def s3_write_json(hook: S3Hook, data: dict, bucket: str, key: str) -> None:
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    buf = io.BytesIO(json_str.encode('utf-8'))
    buf.seek(0)
    hook.load_file_obj(file_obj=buf, key=key, bucket_name=bucket, replace=True) 


# -----------------
# Таски
# -----------------


def init_pipeline(**context):
    start_ts = datetime.utcnow().isoformat()
    _LOG.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="init_pipeline_start", value=start_ts)


def collect_data(**context) -> None:
    import io
    import pandas as pd

    start_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="collect_data_start", value=start_ts)

    _LOG.info(f"Началась загрузка данных")

    from sklearn.datasets import fetch_openml

    titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
    data = pd.concat([titanic.data, titanic.target], axis=1)

    _LOG.info(f"Данные получили")

    s3_hook = S3Hook(AWS_CONN_ID)

    s3_write_csv(s3_hook, data, S3_BUCKET, MY_SURNAME + "/titanic.csv")

    end_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="collect_data_end", value=end_ts)

    _LOG.info(f"Данные успешно загружены")


def split_and_preprocess(**context) -> None:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    _LOG.info(f"Разделить и предобработать данные")
    _LOG.info(f"split_and_preprocess")

    start_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="split_and_preprocess_start", value=start_ts)

    s3_hook = S3Hook(AWS_CONN_ID)
    
    data = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic.csv")

    data_clean = data.dropna(subset=['survived'])
    
    X = data_clean.drop(columns=['survived']).reset_index(drop=True)
    y = data_clean['survived'].reset_index(drop=True)
    
    # Числовые колонки (int, float)
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    cat_cols = X.columns[X.dtypes == 'object']
    
    X_filled = X.copy()
    
    
    # Заполняем пропуски числовые колонки
    if len(num_cols) > 0:  # Используем len() вместо прямой проверки
        num_imputer = SimpleImputer(strategy='median')
        X_filled[num_cols] = num_imputer.fit_transform(X_filled[num_cols])
    
    # Заполняем категориальные колонки
    if len(cat_cols) > 0:  # Используем len() вместо прямой проверки
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_filled[cat_cols] = cat_imputer.fit_transform(X_filled[cat_cols])
    
    X = X_filled.dropna()
    y = y[:X.shape[0]]
    
    
    # обрабатываем категориальные признаки
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    X_cat_encoded = ohe.fit_transform(X[cat_cols])
    
    feature_names = ohe.get_feature_names_out(cat_cols)
    
    X_cat_df = pd.DataFrame(X_cat_encoded, columns=feature_names, index=X.index)
    
    X = pd.concat([X[num_cols], X_cat_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    s3_write_csv(s3_hook, X_train, S3_BUCKET, MY_SURNAME + "/titanic_x_train.csv")
    s3_write_csv(s3_hook, X_test, S3_BUCKET, MY_SURNAME + "/titanic_x_test.csv")
    s3_write_csv(s3_hook, y_train, S3_BUCKET, MY_SURNAME + "/titanic_y_train.csv")
    s3_write_csv(s3_hook, y_test, S3_BUCKET, MY_SURNAME + "/titanic_y_test.csv")

    _LOG.info(f"Xtrain, ytrain, XText, ytext успешно загружены")

    end_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="split_and_preprocess_end", value=end_ts)



def train_model(**context) -> None:
    from sklearn.linear_model import LinearRegression


    start_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="train_model_start", value=start_ts)
    _LOG.info(f"Начали обучение модели start-ts {start_ts}")

    s3_hook = S3Hook(AWS_CONN_ID)
    
    X_train = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic_x_train.csv")
    X_test = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic_x_test.csv")
    y_train = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic_y_train.csv")
    y_test = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic_y_test.csv")

    # Обучить модель
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_pred_df = pd.DataFrame(y_pred)

    end_ts = datetime.utcnow().isoformat()
    _LOG.info(f"Закончили обучение модели end-ts {end_ts}")

    filebuffer = io.BytesIO()
    joblib.dump(model, filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key= f"{MY_SURNAME}/linear_titanic.pkl",
        bucket_name=S3_BUCKET,
        replace=True,
    )
    s3_write_csv(s3_hook, y_pred_df, S3_BUCKET, MY_SURNAME + "/titanic_y_pred.csv")
    end_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="train_model_end", value=end_ts)
    _LOG.info(f"Успешно загрузили модель")


def collect_metrics_model(**context) -> None:
    from sklearn.metrics import mean_squared_error, r2_score
    
    print('collect_metrics_model')
    _LOG.info(f"Начали расчет метрик")
    start_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="collect_metrics_start", value=start_ts)

    s3_hook = S3Hook(AWS_CONN_ID)
    
    X_train = s3_read_csv(s3_hook, S3_BUCKET, MY_SURNAME + "/titanic_x_train.csv")
    X_test = s3_read_csv(s3_hook, S3_BUCKET,  MY_SURNAME + "/titanic_x_test.csv")
    y_train = s3_read_csv(s3_hook, S3_BUCKET,  MY_SURNAME + "/titanic_y_train.csv")
    y_test = s3_read_csv(s3_hook, S3_BUCKET,  MY_SURNAME + "/titanic_y_test.csv")
    y_pred = s3_read_csv(s3_hook, S3_BUCKET,  MY_SURNAME + "/titanic_y_pred.csv")

    metrics = {
        "r_squared": float(r2_score(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred)**0.5),
    }
    print(metrics, "metrics")
    s3_write_json(s3_hook, metrics, S3_BUCKET, S3_KEY_MODEL_METRICS)

    _LOG.info(f"Успешно Завершен расчет метрик")

    end_ts = datetime.utcnow().isoformat()
    context["ti"].xcom_push(key="collect_metrics_end", value=end_ts)



def collect_metrics_pipeline(**context) -> None:

    from datetime import datetime
    from dateutil.parser import parse

    _LOG.info("Начинаем сбор метрик пайплайна")
    
    pipeline_start_dt = parse(context["ti"].xcom_pull(task_ids="init_pipeline", key="init_pipeline_start"))
    train_start_dt = parse(context["ti"].xcom_pull(task_ids="train_model", key="train_model_start"))
    train_end_dt = parse(context["ti"].xcom_pull(task_ids="train_model", key="train_model_end"))
    pipeline_end_dt = parse(context["ti"].xcom_pull(task_ids="collect_metrics_model", key="collect_metrics_end"))
    split_and_preprocess_start_dt = parse(context["ti"].xcom_pull(task_ids="split_and_preprocess", key="split_and_preprocess_start"))
    split_and_preprocess_end_dt = parse(context["ti"].xcom_pull(task_ids="split_and_preprocess", key="split_and_preprocess_end"))

    collect_metrics_start_dt = parse(context["ti"].xcom_pull(task_ids="collect_metrics_model", key="collect_metrics_start"))

    collect_metrics_end_dt = parse(context["ti"].xcom_pull(task_ids="collect_metrics_model", key="collect_metrics_end"))
    
    train_duration = (train_end_dt - train_start_dt).total_seconds()
    total_duration = (pipeline_end_dt - pipeline_start_dt).total_seconds()
    split_duration = (split_and_preprocess_end_dt - split_and_preprocess_start_dt).total_seconds()
    collect_metrics_duration = (collect_metrics_end_dt - collect_metrics_start_dt).total_seconds()

    s3_hook = S3Hook(AWS_CONN_ID)
    
    pipeline_metrics = {
        "pipeline_start": pipeline_start_dt.isoformat(),  # ← конвертируем в строку
        "pipeline_end": pipeline_end_dt.isoformat(),      # ← конвертируем в строку
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_minutes": round(total_duration / 60, 2),
        "split_and_preprocess_start": split_and_preprocess_start_dt.isoformat(),
        "split_and_preprocess_end": split_and_preprocess_end_dt.isoformat(),
        "split_and_preprocess_duration_seconds": round(split_duration, 2),
        "train_start": train_start_dt.isoformat(),
        "train_end": train_end_dt.isoformat(),
        "train_duration_seconds": round(train_duration, 2),
        "train_duration_minutes": round(train_duration / 60, 2),
        "collect_metrics_start": collect_metrics_start_dt.isoformat(),
        "collect_metrics_end": collect_metrics_end_dt.isoformat(),
        "collect_metrics_duration_seconds": round(collect_metrics_duration, 2),
        "author": f"{MY_NAME} {MY_SURNAME}",
    }
    s3_write_json(s3_hook, pipeline_metrics, S3_BUCKET, S3_KEY_PIPELINE_METRICS)

    _LOG.info("Завершаем сбор метрик пайплайна")


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw1",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)
    t3 = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    t4 = PythonOperator(task_id="train_model", python_callable=train_model)
    t5 = PythonOperator(task_id="collect_metrics_model", python_callable=collect_metrics_model)
    t6 = PythonOperator(task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline)

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
