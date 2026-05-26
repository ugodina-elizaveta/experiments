import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import requests
import torch
import torch.nn as nn
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
from transformers import TrainerCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------- Логирование в файл и консоль ----------
log_dir = "/root/thesis/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"exp1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Отправка в Telegram ----------
def send_telegram(message, token="", chat_id=""):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": f"🤖 *QLoRA*\n{message}", "parse_mode": "Markdown"}
        requests.post(url, json=data, timeout=5)
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


# ---------- Метрики ----------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = accuracy_score(labels[mask], predictions[mask])

    loss_fct = nn.CrossEntropyLoss()
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))
    perplexity = torch.exp(loss).item()
    return {"accuracy": acc, "perplexity": perplexity}


# ---------- Колбэк для Telegram (исправленный) ----------
class TelegramCallback(TrainerCallback):
    def __init__(self):
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if state.global_step % 50 == 0 and state.global_step > self.last_step:
            self.last_step = state.global_step
            parts = []
            if "loss" in logs:
                parts.append(f"loss={logs['loss']:.4f}")
            if "learning_rate" in logs:
                parts.append(f"lr={logs['learning_rate']:.2e}")
            if "grad_norm" in logs:
                parts.append(f"grad_norm={logs['grad_norm']:.4f}")
            if parts:
                send_telegram(f"Step {state.global_step}: " + ", ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            msg = f"Evaluation at step {state.global_step}: eval_loss={metrics.get('eval_loss', 0):.4f}, eval_acc={metrics.get('eval_accuracy', 0):.4f}"
            send_telegram(msg)


# ---------- Основная функция ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/thesis/dataset_prepared/tokenized')
    parser.add_argument('--tokenizer_dir', type=str, default='/root/thesis/dataset_prepared/tokenizer')
    parser.add_argument('--output_dir', type=str, default='/root/thesis/experiments/models/qlora_run')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--save_steps', type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК QLoRA (РАБОЧАЯ ВЕРСИЯ)")
    logger.info(f"Параметры: {vars(args)}")
    logger.info("=" * 60)
    send_telegram(
        f"🚀 Запуск: batch={args.batch_size}, grad_acc={args.gradient_accumulation}, lr={args.learning_rate}, max_len={args.max_length}"
    )

    try:
        # Данные
        logger.info("📂 Загрузка данных...")
        data = load_from_disk(args.data_dir)
        train_dataset = data['train']
        val_dataset = data['validation']
        logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        logger.info("🚀 Загрузка модели...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-mini-instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False,
        )
        logger.info(f"✅ Модель загружена. Параметров: {model.num_parameters():,}")

        model = prepare_model_for_kbit_training(model)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        total_steps = len(train_dataset) * args.num_epochs / (args.batch_size * args.gradient_accumulation)
        warmup_steps = int(0.1 * total_steps)
        logger.info(f"Всего шагов: {total_steps:.0f}, Warmup: {warmup_steps}")

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
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=2,
            report_to="none",
            remove_unused_columns=False,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            eval_accumulation_steps=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2), TelegramCallback()],
        )

        send_telegram("🔥 Обучение началось")
        logger.info("🏋️ Начало обучения...")

        trainer.train()

        logger.info("💾 Сохранение модели...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model.save_pretrained(os.path.join(args.output_dir, "adapter"))
        send_telegram("💾 Модель сохранена")

        logger.info("📊 Оценка на тестовой выборке...")
        test_results = trainer.evaluate(data['test'])
        logger.info(f"Test results: {test_results}")
        send_telegram(
            f"📈 Финальные результаты: loss={test_results.get('eval_loss', 0):.4f}, acc={test_results.get('eval_accuracy', 0):.4f}"
        )

        logger.info("✅ Эксперимент успешно завершён!")

    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)
        send_telegram(f"❌ Ошибка: {str(e)[:200]}")
        raise e


if __name__ == "__main__":
    main()
