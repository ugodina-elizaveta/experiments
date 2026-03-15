#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент 1: LoRA (Low-Rank Adaptation) - обычная, не QLoRA
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from exp_logger import ExperimentLogger


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    if len(mask.shape) > len(predictions.shape):
        mask = mask.squeeze()
    acc = accuracy_score(labels[mask], predictions[mask])

    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
    perplexity = torch.exp(loss).item()

    return {"accuracy": acc, "perplexity": perplexity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/thesis/dataset_prepared/tokenized')
    parser.add_argument('--tokenizer_dir', type=str, default='/root/thesis/dataset_prepared/tokenizer')
    parser.add_argument('--output_dir', type=str, default='/root/thesis/experiments/models/lora')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--telegram_token', type=str, default='7504803683:AAEUEb9yplOjOiUsjVZ3GuYXs8ILni8aC-I')
    parser.add_argument('--telegram_chat_id', type=str, default='962369479')
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    args = parser.parse_args()

    os.makedirs('/root/thesis/logs', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger = ExperimentLogger(
        experiment_name="lora-finetuning",
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id,
    )

    try:
        params = {
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "method": "lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "model": "Phi-3.5-mini-instruct",
            "precision": "fp16",
        }
        logger.start_run(params)

        # Загрузка данных
        logger.logger.info("📂 Загрузка данных...")
        data = load_from_disk(args.data_dir)
        train_dataset = data['train']
        val_dataset = data['validation']
        test_dataset = data['test']

        logger.logger.info(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Загрузка модели в fp16
        logger.logger.info("🚀 Загрузка модели Phi-3.5...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.logger.info(f"✅ Модель загружена. Параметров: {model.num_parameters():,}")

        # Конфигурация LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Применяем LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        # Вычисляем шаги
        total_steps = len(train_dataset) * args.num_epochs / (args.batch_size * args.gradient_accumulation)
        warmup_steps = int(0.1 * total_steps)
        logging_steps = max(1, args.eval_steps // 10)

        logger.logger.info(f"📈 Всего шагов: {total_steps:.0f}, Warmup: {warmup_steps}")

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            logging_steps=logging_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=4,
            report_to="mlflow",
            remove_unused_columns=False,
            optim="adamw_torch",
        )

        # Кастомный Trainer
        class LoggingTrainer(Trainer):
            def __init__(self, exp_logger, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.exp_logger = exp_logger

            def log(self, logs, start_time=None):
                super().log(logs, start_time)
                for key, value in logs.items():
                    if key.startswith("eval_"):
                        self.exp_logger.log_metric(key, value, self.state.global_step)
                    elif key in ["loss", "learning_rate"]:
                        self.exp_logger.log_metric(key, value, self.state.global_step)

        trainer = LoggingTrainer(
            exp_logger=logger,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        # Обучение
        logger.send_telegram("🔥 *LoRA обучение началось*")
        logger.logger.info("🏋️ Начало обучения...")
        trainer.train()

        # Сохранение модели
        logger.logger.info("💾 Сохранение модели...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        logger.log_artifact(args.output_dir)
        logger.send_telegram("💾 *LoRA модель сохранена*")

        # Оценка на тесте
        logger.logger.info("📊 Оценка на тестовой выборке...")
        test_results = trainer.evaluate(test_dataset)
        logger.log_metrics({f"test/{k}": v for k, v in test_results.items()})

        results_msg = "📈 *Финальные результаты:*\n"
        for k, v in test_results.items():
            results_msg += f"• {k}: {v:.4f}\n"
        logger.send_telegram(results_msg)

        logger.end_run()

    except Exception as e:
        logger.log_error(str(e))
        logger.end_run(status="FAILED")
        raise e


if __name__ == "__main__":
    main()
