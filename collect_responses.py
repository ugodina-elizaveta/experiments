#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сбор ответов всех моделей для экспертной оценки
"""

import json

import pandas as pd
import requests
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Тестовые запросы (50 примеров из вашей тестовой выборки)
test_queries = [
    "Я постоянно чувствую тревогу, что делать?",
    "Не могу уснуть по ночам, ворочаюсь часами",
    "Чувствую себя одиноко, хотя вокруг есть люди",
    # ... добавьте остальные из вашего test_dataset
]


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained("/root/thesis/dataset_prepared/tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.float16, device_map="auto"
    )
    return model, tokenizer


def load_lora_model(base_model, lora_path):
    model = PeftModel.from_pretrained(base_model, lora_path)
    return model


def generate_response(model, tokenizer, prompt):
    messages = [
        {"role": "system", "content": "Ты — эмпатичный помощник, оказывающий психологическую поддержку."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.9)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Извлекаем только ответ ассистента
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    return response


def call_yandex_gpt(prompt):
    # Ваш код для YandexGPT API
    pass


def main():
    results = []

    # Загружаем все модели
    base_model, tokenizer = load_base_model()
    lora_model = load_lora_model(base_model, "/root/thesis/experiments/models/lora")
    qlora_model = load_lora_model(base_model, "/root/thesis/experiments/models/qlora/adapter")

    for query in tqdm(test_queries):
        row = {"query": query}

        # Base model
        row["base_response"] = generate_response(base_model, tokenizer, query)

        # LoRA
        row["lora_response"] = generate_response(lora_model, tokenizer, query)

        # QLoRA
        row["qlora_response"] = generate_response(qlora_model, tokenizer, query)

        # YandexGPT
        # row["yandex_response"] = call_yandex_gpt(query)

        results.append(row)

    # Сохраняем
    df = pd.DataFrame(results)
    df.to_csv("model_responses_for_experts.csv", index=False)

    # Создаем форму для экспертов
    create_expert_form(df)


if __name__ == "__main__":
    main()
