"""
Fine-tuning script for facebook/nllb-200-distilled-600M model.
"""

import os
import json
from typing import Dict, Tuple
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate
from settings.config import settings


def load_translations() -> Tuple[Dataset, Dataset]:
    """Load translations from JSON file and split into train/validation sets."""
    file_path = os.path.join(settings.training.output_dir, "data", "translations.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split data into train and validation sets
    train_data, val_data = train_test_split(
        data["translations"],
        test_size=0.2,  # 20% for validation
        random_state=42,  # For reproducibility
    )

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """Preprocess the datasets."""
    inputs = examples["translation"][settings.training.source_lang]
    targets = examples["translation"][settings.training.target_lang]

    model_inputs = tokenizer(
        inputs,
        max_length=settings.model.max_length,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=settings.model.max_length,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds) -> Dict:
    """Compute BLEU score for evaluation."""
    metric = evaluate.load("sacrebleu")

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def main():
    # Load tokenizer and model
    global tokenizer  # Make tokenizer available to compute_metrics
    tokenizer = AutoTokenizer.from_pretrained(settings.model.name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.model.name)

    # Load and split datasets
    train_dataset, eval_dataset = load_translations()

    # Preprocess datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=settings.training.output_dir,
        evaluation_strategy=settings.training.evaluation_strategy,
        learning_rate=settings.training.learning_rate,
        per_device_train_batch_size=settings.training.per_device_train_batch_size,
        per_device_eval_batch_size=settings.training.per_device_eval_batch_size,
        num_train_epochs=settings.training.num_train_epochs,
        weight_decay=settings.training.weight_decay,
        save_total_limit=settings.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=settings.training.metric_for_best_model,
        report_to="tensorboard",
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()


if __name__ == "__main__":
    main()
