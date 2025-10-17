import os
import logging
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_breast_cancer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MY_NAME = "VLADISLAV"
MY_SURNAME = "KUDRYAKOV"
EXPERIMENT_NAME = f"{MY_SURNAME}_V_HW2"  # Исправлено имя
PARENT_RUN_NAME = "vladislav747"  # ник в телеграм
MODEL_NAME = f"LogReg_{MY_SURNAME}"
MLFLOW_TRACKING_URI = "http://localhost:5050"  # MLFlow в Docker

# Исправленные модели для классификации
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}



def setup_mlflow():
    """Настройка MLFlow: подключение и создание эксперимента с обработкой удаленных"""
    logger.info("Настраиваем MLFlow...")
    
    # Подключение к MLFlow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    # Создание или получение эксперимента
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            # Эксперимента нет - создаем новый
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Создан новый эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
        elif experiment.lifecycle_stage == "deleted":
            # Эксперимент удален - восстанавливаем или создаем новый
            logger.warning(f"Эксперимент {EXPERIMENT_NAME} был удален")
            try:
                # Пытаемся восстановить
                client.restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
                logger.info(f"Восстановлен эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
            except Exception as restore_error:
                # Если не получилось восстановить - создаем новый с другим именем
                new_name = f"{EXPERIMENT_NAME}_v2"
                experiment_id = mlflow.create_experiment(new_name)
                logger.info(f"Создан новый эксперимент: {new_name} (ID: {experiment_id})")
                # Обновляем глобальную переменную
                global EXPERIMENT_NAME
                EXPERIMENT_NAME = new_name
        else:
            # Эксперимент существует и активен
            experiment_id = experiment.experiment_id
            logger.info(f"Используем существующий эксперимент: {EXPERIMENT_NAME} (ID: {experiment_id})")
            
    except Exception as e:
        logger.error(f"Ошибка при работе с экспериментом: {e}")
        # Создаем эксперимент с уникальным именем
        import time
        unique_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
        experiment_id = mlflow.create_experiment(unique_name)
        logger.info(f"Создан резервный эксперимент: {unique_name} (ID: {experiment_id})")
        global EXPERIMENT_NAME
        EXPERIMENT_NAME = unique_name
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def prepare_data():
    """Подготовка данных: загрузка, препроцессинг, разделение"""
    logger.info("Начинаем подготовку данных...")
    
    # Загружаем датасет
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    logger.info(f"Загружен датасет: {X.shape[0]} образцов, {X.shape[1]} признаков")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Разделение данных: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


def train_and_log(name, model, X_train, y_train, X_test, y_test):
    """Обучение одной модели с логированием в MLFlow"""
    logger.info(f"Обучаем модель: {name}")
    
    with mlflow.start_run(run_name=name, nested=True):
        # Обучить модель
        model.fit(X_train, y_train)

        # Сделать predict
        y_pred = model.predict(X_test)
        
        # Вычисление метрик
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Логирование метрик
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        
        # Логирование параметров модели
        params = model.get_params()
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Получить описание данных
        signature = infer_signature(X_train, y_pred)
        input_example = X_train.iloc[:5]  # Первые 5 строк как пример

        # Сохранить модель в артифактори
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"Модель {name} обучена. F1-score: {f1:.4f}")
        
        # Возвращаем F1-score для сравнения
        return f1, mlflow.active_run().info.run_id


def select_and_register_best_model(results):
    """Выбор лучшей модели и регистрация в MLFlow Registry"""
    logger.info("Выбираем лучшую модель...")
    
    # Находим лучшую модель по F1-score
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
    best_result = results[best_model_name]
    best_f1 = best_result['f1_score']
    best_run_id = best_result['run_id']
    
    logger.info(f"Лучшая модель: {best_model_name} с F1-score: {best_f1:.4f}")
    
    # Регистрируем лучшую модель
    model_uri = f"runs:/{best_run_id}/model"
    
    try:
        # Регистрируем модель
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )
        
        # Переводим в стадию Staging
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging"
        )
        
        logger.info(f"Модель {MODEL_NAME} v{registered_model.version} зарегистрирована и переведена в Staging")
        logger.info(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с F1-score: {best_f1:.4f}")
        
        return best_model_name, best_f1, registered_model.version
        
    except Exception as e:
        logger.error(f"Ошибка при регистрации модели: {e}")
        raise


def main():
    """Основная функция запуска эксперимента"""
    try:
        logger.info("🚀 Запуск ML-эксперимента с MLFlow")
        
        # 1. Настройка MLFlow
        experiment_id = setup_mlflow()
        
        # 2. Подготовка данных
        X_train, X_test, y_train, y_test = prepare_data()
        
        # 3. Обучение всех моделей в parent run
        results = {}
        
        with mlflow.start_run(run_name=PARENT_RUN_NAME):
            logger.info(f"Запущен parent run: {PARENT_RUN_NAME}")
            
            # Логируем общие параметры эксперимента
            mlflow.log_param("dataset", "breast_cancer")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("n_models", len(models))
            
            # Обучаем каждую модель
            for model_name, model in models.items():
                f1_score_value, run_id = train_and_log(
                    model_name, model, X_train, y_train, X_test, y_test
                )
                results[model_name] = {
                    'f1_score': f1_score_value,
                    'run_id': run_id
                }
            
            # Логируем сводные метрики в parent run
            best_f1 = max(results.values(), key=lambda x: x['f1_score'])['f1_score']
            avg_f1 = np.mean([r['f1_score'] for r in results.values()])
            
            mlflow.log_metric("best_f1_score", best_f1)
            mlflow.log_metric("average_f1_score", avg_f1)
            
            logger.info(f"Parent run завершен. Лучший F1-score: {best_f1:.4f}")
        
        # 4. Выбор и регистрация лучшей модели
        best_model_name, best_f1, model_version = select_and_register_best_model(results)
        
        logger.info("✅ Эксперимент успешно завершен!")
        logger.info(f"📊 Результаты всех моделей:")
        for name, result in results.items():
            logger.info(f"   {name}: F1-score = {result['f1_score']:.4f}")
        
        logger.info(f"🥇 Победитель: {best_model_name} (F1-score: {best_f1:.4f})")
        logger.info(f"📝 Зарегистрирована как: {MODEL_NAME} v{model_version} в стадии Staging")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в эксперименте: {e}")
        raise


if __name__ == "__main__":
    main()