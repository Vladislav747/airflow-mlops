import os
import mlflow
import logging
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MY_NAME = "VLADISLAV"
MY_SURNAME = "KUDRYAKOV"
PARENT_RUN_NAME = "Vladislav747"
EXPERIMENT_NAME = f"{MY_SURNAME}_{MY_NAME[0]}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

models = dict(zip(["RandomForest", "LogisticRegression", "HistGB"], 
                  [RandomForestClassifier(), LogisticRegression(), GradientBoostingClassifier()]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def prepare_data():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    logger.info(f"Загружен датасет: {X.shape[0]} образцов, {X.shape[1]} признаков")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Разделение данных: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_and_log(name, model, X_train, y_train, X_test, y_test):
    logger.info(f"Обучаем модель: {name}")
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    f1 = f1_score(y_test, prediction, average='weighted')
    accuracy = accuracy_score(y_test, prediction)

    params = model.get_params()
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

    signature = infer_signature(X_test, prediction)
    
    # Логируем метрики
    mlflow.log_metric('f1_score', f1)
    mlflow.log_metric('accuracy', accuracy)

    # Создаем input_example для лучшего понимания модели
    input_example = X_train.iloc[:5]
    
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # Явно указываем путь в артифактори
        signature=signature,
        input_example=input_example
    )

    eval_data = X_test.copy()
    target_col = "target"
    eval_data[target_col] = y_test.values if hasattr(y_test, 'values') else y_test

    mlflow.evaluate(
        model_info.model_uri,
        data=eval_data,
        targets=target_col,
        model_type="classifier",
        evaluators=["default"]
    )

    logger.info(f"Модель {name} обучена. F1-score: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    return f1, mlflow.active_run().info.run_id

def select_and_register_best_model(results):
    logger.info("Выбираем лучшую модель...")
    
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_result = results[best_model_name]
    best_f1 = best_result['f1_score']
    best_run_id = best_result['run_id']

    BEST_MODEL_NAME = f"{best_model_name}_{MY_SURNAME}"
    
    logger.info(f"Лучшая модель: {best_model_name} с F1-score: {best_f1:.4f}")
    
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=BEST_MODEL_NAME
        )
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=BEST_MODEL_NAME,
            version=registered_model.version,
            stage="Staging"
        )
        
        logger.info(f"Модель {BEST_MODEL_NAME} v{registered_model.version} зарегистрирована и переведена в Staging")
        logger.info(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с F1-score: {best_f1:.4f}")
        
    except Exception as e:
        logger.error(f"Ошибка при регистрации модели: {e}")
        raise

def main():
    global EXPERIMENT_NAME
    logger.info("Запускаем MLFlow эксперимент")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Создан новый эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
        elif experiment.lifecycle_stage == "deleted":
            # Эксперимент в статусе deleted - восстанавливаем или создаем новый
            logger.warning(f"Эксперимент {EXPERIMENT_NAME} был удален")
            try:
                client.restore_experiment(experiment.experiment_id)
                logger.info(f"Восстановлен эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
            except Exception as restore_error:
                logger.error(f"Ошибка при восстановлении эксперимента: {restore_error}")
                new_name = f"{EXPERIMENT_NAME}"
                experiment_id = mlflow.create_experiment(new_name)
                logger.info(f"Создан новый эксперимент: {new_name} (ID: {experiment_id})")
                EXPERIMENT_NAME = new_name
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Используем существующий эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
    except Exception as e:
        logger.error(f"Ошибка в эксперименте: {e}")
        raise

    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = prepare_data()

    results = {}

    with mlflow.start_run(run_name=PARENT_RUN_NAME, experiment_id=experiment_id, description="Parent run") as parent_run:
        mlflow.log_param("dataset", "breast_cancer")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_models", len(models))
        
        for model in models.keys():
            with mlflow.start_run(run_name=model, experiment_id=experiment_id, nested=True) as child_run:
                f1_score_value, run_id = train_and_log(
                    model, models[model], X_train, y_train, X_test, y_test
                )
                results[model] = {
                    'f1_score': f1_score_value,
                    'run_id': run_id
                }
        best_f1 = max(results.values(), key=lambda x: x['f1_score'])['f1_score']
        avg_f1 = np.mean([r['f1_score'] for r in results.values()])
        
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_metric("average_f1_score", avg_f1)

        select_and_register_best_model(results)
        
        logger.info(f"Parent run завершен. Лучший F1-score: {best_f1:.4f}")
   


if __name__ == "__main__":
    main()