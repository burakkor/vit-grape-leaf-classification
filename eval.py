import numpy as np
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from dataset import Leaves, collate_fn

CONFIG = {
    "model_path": "models/checkpoints/checkpoint-9",
    "processor_name": "google/vit-base-patch16-224",
    "csv_path": "dataset/annotations.csv",
    "batch_size": 256,
}

metric = evaluate.load("accuracy")

def compute_metrics(all_preds, all_labels):
    return metric.compute(predictions=all_preds, references=all_labels)


model = AutoModelForImageClassification.from_pretrained(CONFIG["model_path"])
processor = AutoImageProcessor.from_pretrained(CONFIG["processor_name"])

dataset_test = Leaves(processor, csv_path=CONFIG["csv_path"], split="test")
dataloader = DataLoader(dataset_test, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch["pixel_values"])
        preds = np.argmax(outputs.logits.numpy(), axis=1)
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

results = compute_metrics(all_preds, all_labels)
print(results)
