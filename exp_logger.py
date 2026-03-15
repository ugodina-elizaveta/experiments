# /root/thesis/experiments/exp_logger.py
import logging
import requests
import json
from datetime import datetime
import mlflow
import mlflow.pytorch
from pathlib import Path


class ExperimentLogger:
    def __init__(
        self,
        experiment_name,
        telegram_token="7504803683:AAEUEb9yplOjOiUsjVZ3GuYXs8ILni8aC-I",  # ваш токен
        telegram_chat_id="962369479",  # ваш chat_id
        mlflow_tracking_uri="http://localhost:5000",
    ):

        self.experiment_name = experiment_name
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_url = f"https://api.telegram.org/bot{telegram_token}"

        # Настройка MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        # Локальный логгер
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            f'/root/thesis/logs/{experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        # Добавляем вывод в консоль
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)

        self.start_time = datetime.now()
        self.current_step = 0

    def send_telegram(self, message):
        """Отправляет сообщение в Telegram"""
        try:
            url = f"{self.telegram_url}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": f"🤖 *{self.experiment_name}*\n{message}",
                "parse_mode": "Markdown",
            }
            requests.post(url, json=data, timeout=5)
        except Exception as e:
            self.logger.warning(f"Не удалось отправить в Telegram: {e}")

    def start_run(self, params=None):
        """Начинает новый запуск в MLflow"""
        self.run = mlflow.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%H%M%S')}")
        if params:
            mlflow.log_params(params)
        self.send_telegram(f"🚀 *Эксперимент запущен*\nПараметры: {json.dumps(params, indent=2)}")
        self.logger.info(f"Эксперимент {self.experiment_name} запущен")

    def log_params(self, params):
        """Логирует параметры"""
        mlflow.log_params(params)
        self.logger.info(f"Параметры: {params}")

    def log_metric(self, name, value, step=None):
        """Логирует метрику"""
        mlflow.log_metric(name, value, step=step)
        self.current_step = step if step else self.current_step + 1

        # Важные метрики отправляем в Telegram
        if name in ['eval_loss', 'eval_accuracy', 'eval_perplexity']:
            self.send_telegram(f"📊 *{name}* = {value:.4f} (step {self.current_step})")

    def log_metrics(self, metrics, step=None):
        """Логирует несколько метрик"""
        mlflow.log_metrics(metrics, step=step)
        for name, value in metrics.items():
            if name in ['eval_loss', 'eval_accuracy', 'eval_perplexity']:
                self.send_telegram(f"📊 *{name}* = {value:.4f}")

    def log_artifact(self, local_path):
        """Сохраняет файл в артефакты"""
        mlflow.log_artifact(local_path)

    def log_model(self, model, artifact_path):
        """Сохраняет модель"""
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self, status="FINISHED"):
        """Завершает запуск"""
        duration = datetime.now() - self.start_time
        mlflow.end_run(status=status)
        self.send_telegram(f"✅ *Эксперимент завершен*\n⏱️ Время: {duration}")
        self.logger.info(f"Эксперимент завершен. Длительность: {duration}")

    def log_error(self, error_msg):
        """Логирует ошибку"""
        self.logger.error(error_msg)
        self.send_telegram(f"❌ *ОШИБКА*\n{error_msg}")
