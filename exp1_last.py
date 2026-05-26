
"""
Эксперимент 1: Last-layer fine-tuning (дообучение только последних 2 слоёв)
"""

import os
import sys
import logging
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_from_disk
import wandb
import numpy as np
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/exp1_last_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = accuracy_score(labels[mask], predictions[mask])
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(torch.tensor(logits).view(-1, logits.shape[-1]), torch.tensor(labels).view(-1))
    perplexity = torch.exp(loss).item()
    return {"accuracy": acc, "perplexity": perplexity}


def freeze_except_last_layers(model, num_layers_to_unfreeze=2):
    """Заморозить все параметры, кроме последних num_layers_to_unfreeze слоёв и lm_head"""
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Размораживаем последние слои трансформера
    transformer_layers = model.model.layers
    for layer in transformer_layers[-num_layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Размораживаем lm_head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Также можно разморозить нормализацию перед выходом
    if hasattr(model.model, 'norm'):
        for param in model.model.norm.parameters():
            param.requires_grad = True

    # Подсчёт обучаемых параметров
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Обучаемых параметров: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/thesis/dataset_prepared/tokenized')
    parser.add_argument('--tokenizer_dir', type=str, default='/root/thesis/dataset_prepared/tokenizer')
    parser.add_argument('--output_dir', type=str, default='/root/thesis/experiments/models/last_layer')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_last_layers', type=int, default=2, help='Количество последних слоёв для дообучения')
    parser.add_argument('--wandb_project', type=str, default='YourLifePilot-Thesis')
    parser.add_argument('--wandb_run', type=str, default='last-layer')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    logger.info("Загрузка данных...")
    data = load_from_disk(args.data_dir)
    train_dataset = data['train']
    val_dataset = data['validation']
    test_dataset = data['test']

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Загрузка модели Phi-3.5...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    # Заморозка всех слоёв, кроме последних
    freeze_except_last_layers(model, num_layers_to_unfreeze=args.num_last_layers)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=args.wandb_run,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Начало обучения Last-layer...")
    trainer.train()

    # Сохраняем модель
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Модель сохранена в {args.output_dir}")

    test_results = trainer.evaluate(test_dataset)
    logger.info(f"Test results: {test_results}")
    wandb.log({"test/" + k: v for k, v in test_results.items()})

    wandb.finish()
    logger.info("Эксперимент Last-layer завершен!")


if __name__ == "__main__":
    main()
