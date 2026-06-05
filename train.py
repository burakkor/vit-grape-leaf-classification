import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from dataset import Leaves, collate_fn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import numpy as np

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

BATCH_SIZE = 32

dataset_train = Leaves(processor, csv_path="dataset/annotations.csv", split="train")
dataset_valid = Leaves(processor, csv_path="dataset/annotations.csv", split="valid")

model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224",
                                                        label2id=dataset_train.name2idx,
                                                        id2label=dataset_train.idx2name,
                                                        ignore_mismatched_sizes=True)

# dataloader_train = DataLoader(dataset_train,
#                         batch_size=BATCH_SIZE,
#                         collate_fn=collate_fn)

# dataloader_valid = DataLoader(dataset_valid,
#                         batch_size=BATCH_SIZE,
#                         collate_fn=collate_fn)


args = TrainingArguments(
    output_dir=r"models/no_grape/",
    remove_unused_columns=False,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

import evaluate

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
    tokenizer=processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()
