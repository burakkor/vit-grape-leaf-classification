from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import Dataset, DataLoader
from dataset import Leaves, collate_fn
import evaluate
import numpy as np

metric = evaluate.load("accuracy")
def compute_metrics(predictions, labels):
    predictions = np.argmax(predictions['logits'].detach(), axis=1)

    return metric.compute(predictions=predictions, references=labels)


model = AutoModelForImageClassification.from_pretrained(pretrained_model_name_or_path=r"models\no_grape\checkpoint-9")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

dataset_test = Leaves(processor, csv_path="dataset/annotations.csv", split="test")
dataloader = DataLoader(dataset_test, batch_size=256, shuffle=True)
batch = next(iter(dataloader))


preds = model(batch['pixel_values'])
labels = batch['labels']
print(compute_metrics(preds, labels))

# Without grape-leaf disease dataset: 79% accuracy
# With all three datasets combined: 92% accuracy
# Note: the higher accuracy with grape dataset may be due to its white background making classification easier
