import numpy as np
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from dataset import Leaves, collate_fn

CONFIG = {
    "model_name": "google/vit-base-patch16-224",
    "csv_path": "dataset/annotations.csv",
    "output_dir": "models/checkpoints/",
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_train_epochs": 3,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
}

processor = AutoImageProcessor.from_pretrained(CONFIG["model_name"])

dataset_train = Leaves(processor, csv_path=CONFIG["csv_path"], split="train")
dataset_valid = Leaves(processor, csv_path=CONFIG["csv_path"], split="valid")

model = AutoModelForImageClassification.from_pretrained(
    CONFIG["model_name"],
    label2id=dataset_train.name2idx,
    id2label=dataset_train.idx2name,
    ignore_mismatched_sizes=True,
)

args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["num_train_epochs"],
    warmup_ratio=CONFIG["warmup_ratio"],
    logging_steps=CONFIG["logging_steps"],
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=dataset_train,
    eval_dataset=dataset_valid,
    processing_class=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
