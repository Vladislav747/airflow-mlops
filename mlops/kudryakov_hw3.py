import io
import os
import logging
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "s3_connection"
MY_NAME = "VLADISLAV"
MY_SURNAME = "KUDRYAKOV"
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}{MY_NAME[0]}_Final"
TELEGRAM_NAME="Vladislav747"

S3_BUCKET = Variable.get("S3_BUCKET")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = "ru-central1"
AWS_ENDPOINT_URL = "https://storage.yandexcloud.net"



S3_KEY_RAW_DATA = "raw_data/wine_dataset.csv"
S3_KEY_X_TRAIN = "processed_data/x_train.csv"
S3_KEY_X_TEST = "processed_data/x_test.csv"
S3_KEY_Y_TRAIN = "processed_data/y_train.csv"
S3_KEY_Y_TEST = "processed_data/y_test.csv"


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


def s3_write_pickle(hook: S3Hook, obj, bucket: str, key: str) -> None:
    """Запись объекта в pickle файл в S3"""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)


def s3_read_pickle(hook: S3Hook, bucket: str, key: str):
    """Чтение объекта из pickle файла в S3"""
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pickle.load(buf)


# -----------------
# Таски
# -----------------
def init_pipeline(**context):
    ts = datetime.utcnow().isoformat()
    logging.info(f"pipeline_start={ts}")
    context['ti'].xcom_push(key='pipeline_start_time', value=ts)

    return ts


def collect_data(**context):
    logging.info("Началась загрузка данных")
    try:
        wine_data = load_wine()
        df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
        df['target'] = wine_data.target
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        s3_write_csv(s3_hook, df, S3_BUCKET, S3_KEY_RAW_DATA)

        context['ti'].xcom_push(key='data_shape', value=df.shape)
        context['ti'].xcom_push(key='data_columns', value=list(df.columns))
        
        logging.info("Сырые данные успешно сохранены в S3")
        
    except Exception as e:
        logging.info(f"Ошибка при загрузке данных: {e}")
        raise




def split_and_preprocess(**context):
    logging.info("Обработка данных")
    try:
        # Читаем данные из S3
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        df = s3_read_csv(s3_hook, S3_BUCKET, S3_KEY_RAW_DATA)

        logging.info(f"Загружены данные из бакета S3: {df.shape}")

        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

         # Минимальная обработка данных
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())


         # 2. Стандартизация признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        train_data = X_train_scaled.copy()
        train_data['target'] = y_train
        test_data = X_test_scaled.copy()
        test_data['target'] = y_test
        
        s3_write_csv(s3_hook, train_data, S3_BUCKET, S3_KEY_X_TRAIN)
        s3_write_csv(s3_hook, test_data, S3_BUCKET, S3_KEY_X_TEST)
        s3_write_csv(s3_hook, train_data, S3_BUCKET, S3_KEY_Y_TRAIN)
        s3_write_csv(s3_hook, test_data, S3_BUCKET, S3_KEY_Y_TEST)

        s3_write_pickle(s3_hook, scaler, S3_BUCKET, "models/scaler.pkl")

        context['ti'].xcom_push(key='train_shape', value=X_train_scaled.shape)
        context['ti'].xcom_push(key='test_shape', value=X_test_scaled.shape)

        logging.info("Обработанные данные успешно сохранены в S3")

    except Exception as e:
        logging.error(f"Ошибка при чтении и обработке данных из S3 - split_and_preprocess: {e}")
        raise


def train_and_log_mlflow(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    **context,
):
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
    os.environ["AWS_S3_ADDRESSING_STYLE"] = "path"
    os.environ["AWS_S3_SIGNATURE_VERSION"] = "s3v4"
    
    logging.info(f"AWS настройки: endpoint={AWS_ENDPOINT_URL}, region={AWS_DEFAULT_REGION}")
    logging.info("AWS credentials configured for MLflow")

    logging.info("Старт обучения модели - train_and_log_mlflow")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
                logging.info(f"Создан новый эксперимент: {MLFLOW_EXPERIMENT_NAME}")
            else:
                experiment_id = experiment.experiment_id
                logging.info(f"Используем существующий эксперимент: {MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            logging.info(f"Создан новый эксперимент: {MLFLOW_EXPERIMENT_NAME}")
            experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        
        train_data = s3_read_csv(s3_hook, S3_BUCKET, S3_KEY_X_TRAIN)
        test_data = s3_read_csv(s3_hook, S3_BUCKET, S3_KEY_X_TEST)


        X_train = train_data.drop('target', axis=1)
        y_train = train_data['target']
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        logging.info(f"Loaded training data: {X_train.shape}, test data: {X_test.shape}")
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=100)
        }

        best_model = None
        best_score = 0
        best_run_id = None
        model_results = {}


        with mlflow.start_run(run_name=TELEGRAM_NAME) as parent_run:
            logging.info(f"Started parent run: {parent_run.info.run_id}")


            for model_name, model in models.items():
                with mlflow.start_run(run_name=f"{model_name}_{TELEGRAM_NAME}", nested=True) as child_run:
                    logging.info(f"Обучаем {model_name}...")


                    model.fit(X_train, y_train)
                            
                    # Предсказания
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    y_pred_proba_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    precision = precision_score(y_test, y_pred_test, average='weighted')
                    recall = recall_score(y_test, y_pred_test, average='weighted')
                    f1 = f1_score(y_test, y_pred_test, average='weighted')


                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        for param_name, param_value in params.items():
                            mlflow.log_param(param_name, param_value)
                
                    mlflow.log_metric("train_accuracy", train_accuracy)
                    mlflow.log_metric("test_accuracy", test_accuracy)
                    mlflow.log_metric("precision", precision)
                    mlflow.log_metric("recall", recall)
                    mlflow.log_metric("f1_score", f1)
                    
                    logging.info(f"Создаем signature для модели {model_name}")
                    signature = infer_signature(X_train, y_pred_test)
                    input_example = X_test.iloc[:5]

                    try:
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="model",
                            signature=signature,
                            input_example=input_example
                        )
                        logging.info(f"Модель {model_name} успешно залогирована в MLflow")
                    except Exception as e:
                        logging.warning(f"Не удалось залогировать модель {model_name}: {str(e)}")
                    
                    # Сохраняем результаты
                    model_results[model_name] = {
                        'model': model,
                        'test_accuracy': test_accuracy,
                        'f1_score': f1,
                        'run_id': child_run.info.run_id
                    }

                    logging.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}")
                    
                    # Проверяем, лучшая ли это модель (по F1 score)
                    if f1 > best_score:
                        best_score = f1
                        best_model = model_name
                        best_run_id = child_run.info.run_id

            mlflow.log_param("best_model", best_model)
            mlflow.log_metric("best_f1_score", best_score)
            mlflow.log_metric("best_test_accuracy", model_results[best_model]['test_accuracy'])
        
        logging.info(f"Best model: {best_model} with F1 score: {best_score:.4f}")
        
        model_name_registry = f"{best_model}_{MY_SURNAME}"
        
        try:
            best_model_uri = f"runs:/{best_run_id}/model" 
            
            registered_model = mlflow.register_model(
                model_uri=best_model_uri,
                name=model_name_registry
            )
            
            # Переводим в стадию Staging
            client.transition_model_version_stage(
                name=model_name_registry,
                version=registered_model.version,
                stage="Staging"
            )
            
            logging.info(f"Model {model_name_registry} v{registered_model.version} moved to Staging")
            registered_version = registered_model.version
            
        except Exception as e:
            logging.warning(f"Не удалось зарегистрировать модель: {str(e)}")
            registered_version = "1"  # Заглушка
        
        context['ti'].xcom_push(key='best_run_id', value=best_run_id)
        context['ti'].xcom_push(key='best_model_name', value=best_model)
        context['ti'].xcom_push(key='best_f1_score', value=best_score)
        context['ti'].xcom_push(key='registered_model_name', value=model_name_registry)
        context['ti'].xcom_push(key='registered_model_version', value=registered_version)
        
        return f"Model training completed. Best model: {best_model} (F1: {best_score:.4f})"

    except Exception as e:
        logging.error(f"Ошибка при обучении модели - train_and_log_mlflow: {e}")
        raise


def serve_model(**context):
    logging.info("Сервировка модели")
    
    try:
        # Получаем информацию о лучшей модели из XCom
        ti = context['ti']
        best_run_id = ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_run_id')
        best_model_name = ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_model_name')
        best_f1_score = ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_f1_score')
        registered_model_name = ti.xcom_pull(task_ids='train_and_log_mlflow', key='registered_model_name')
        registered_model_version = ti.xcom_pull(task_ids='train_and_log_mlflow', key='registered_model_version')
        
        logging.info(f"Serving model: {best_model_name}")
        logging.info(f"Run ID: {best_run_id}")
        logging.info(f"F1 Score: {best_f1_score}")
        logging.info(f"Registered as: {registered_model_name} v{registered_model_version}")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
        os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
        os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION
        
        s3_hook = S3Hook(aws_conn_id=AWS_CONN_ID)
        test_data = s3_read_csv(s3_hook, S3_BUCKET, S3_KEY_X_TEST)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        
        logging.info(f"Загрузили тестовые данные - serve_model: {X_test.shape}")
        
        # Правильный способ - Загрузка модели из MLflow Model Registry
        model = None
        model_source = "unknown"
        
        try:
            model_uri = f"models:/{registered_model_name}/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            model_source = "MLflow_Registry"
            logging.info("Модель успешно загружена из MLflow Model Registry - serve_model")
        except Exception as e:
            logging.warning(f"Не смогли загрузить модель из MLflow Registry - serve_model: {str(e)}")
        
        # Попытка 2: Еще 1 способ Загрузка модели из конкретного run если первый способ не прошел
        if model is None:
            try:
                model_uri = f"runs:/{best_run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                model_source = "MLflow_Run"
                logging.info("Модель успешно загружена из MLflow Run")
            except Exception as e:
                logging.warning(f"Не смогли загрузить модель из MLflow Run: {str(e)}")
            
            model.fit(X_train, y_train)
            model_source = "Demo_Model"
        
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        final_accuracy = accuracy_score(y_test, predictions)
        final_f1 = f1_score(y_test, predictions, average='weighted')
        final_precision = precision_score(y_test, predictions, average='weighted')
        final_recall = recall_score(y_test, predictions, average='weighted')
        
        logging.info(f"Финальные метрики модели:")
        logging.info(f"- Accuracy: {final_accuracy:.4f}")
        logging.info(f"- F1 Score: {final_f1:.4f}")
        logging.info(f"- Precision: {final_precision:.4f}")
        logging.info(f"- Recall: {final_recall:.4f}")
        
        results_df = pd.DataFrame({
            'true_labels': y_test,
            'predictions': predictions
        })
        
        if probabilities is not None:
            for i in range(probabilities.shape[1]):
                results_df[f'prob_class_{i}'] = probabilities[:, i]
        
        s3_write_csv(s3_hook, results_df, S3_BUCKET, "results/model_predictions.csv")
        
        report = {
            'serving_timestamp': datetime.utcnow().isoformat(),
            'original_best_model': best_model_name,
            'original_best_f1': best_f1_score,
            'best_run_id': best_run_id,
            'registered_model_name': registered_model_name,
            'registered_model_version': registered_model_version,
            'model_source': model_source,
            'final_accuracy': final_accuracy,
            'final_f1_score': final_f1,
            'final_precision': final_precision,
            'final_recall': final_recall,
            'test_samples': len(X_test),
            'prediction_classes': len(np.unique(predictions)),
            'model_type': type(model).__name__
        }
        
        report_df = pd.DataFrame([report])
        s3_write_csv(s3_hook, report_df, S3_BUCKET, "results/model_serving_report.csv")
        
        from sklearn.metrics import classification_report
        class_report = classification_report(y_test, predictions, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        s3_write_csv(s3_hook, class_report_df, S3_BUCKET, "results/classification_report.csv")
        
        logging.info("Сервировка модели завершена успешно!")
        logging.info("Результаты сохранены в S3:")
        
        
    except Exception as e:
        logging.error(f"Error in model serving: {str(e)}")
        raise


default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}

with DAG(
    dag_id="hw33",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    init_pipeline = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
        provide_context=True,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": AWS_DEFAULT_REGION,
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
        },
    )
    serve_model = PythonOperator(
        task_id="serve_model", 
        python_callable=serve_model,
        provide_context=True
    )

    init_pipeline >> collect_data >> split_and_preprocess >> train_and_log_mlflow >> serve_model
