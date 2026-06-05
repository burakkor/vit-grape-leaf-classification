# ViT Plant Disease Classification

Binary classification of grape leaf diseases (healthy vs. mildew) using a fine-tuned Vision Transformer (ViT).

## What It Does

This project fine-tunes a pretrained `google/vit-base-patch16-224` model on a multi-source grape leaf image dataset to classify leaves as either **healthy** or **mildew-infected** (covering both powdery and downy mildew).

Three public datasets were combined and evaluated under different configurations:

| Configuration | Accuracy |
|---|---|
| Without grape-leaf disease dataset | 79% |
| All three datasets combined | 92%* |

*The accuracy boost with the third dataset may be partially explained by its white background making classification easier rather than genuine generalization improvement.

## Project Structure

```
vit-plant-disease-classification/
├── train.py          # Fine-tuning with HuggingFace Trainer API
├── eval.py           # Evaluation on test split
├── attention.ipynb   # Attention map visualization
└── README.md
```

## Datasets

- [Hermos Dataset](https://data.mendeley.com/datasets/j4xs3kh3fd/2) — powdery and downy mildew labels merged into a single `mildew` class
- Mildew-labeled grape leaf images (private collection)
- [Grape-leaf disease dataset](https://universe.roboflow.com/tru-projects-cqcql/grape-leaf-disease-dataset/dataset/1) — only healthy samples used

The merged annotation file (`dataset/annotations.csv`) contains columns: `filename`, `healthy`, `mildew`.

Expected directory layout:

```
dataset/
├── annotations.csv
├── train/
├── valid/
└── test/
```

## Requirements

```
torch
transformers
evaluate
Pillow
opencv-python
pandas
numpy
```

Install with:

```bash
pip install torch transformers evaluate Pillow opencv-python pandas numpy
```

## Usage

**1. Train**

```bash
python train.py
```

Checkpoints are saved to `models/no_grape/` at the end of each epoch. Best model is loaded automatically based on validation accuracy.

**2. Evaluate**

```bash
python eval.py
```

Loads the checkpoint at `models/no_grape/checkpoint-9` and reports accuracy on the test split.

**3. Attention visualization**

Open `attention.ipynb` to visualize attention maps from any layer of the fine-tuned model. Set the image path in the `CONFIG` dict at the top of the relevant cell.

## Key Design Decisions

- **Label merging:** Powdery mildew and downy mildew are merged into a single `mildew` class to simplify the task to binary classification.
- **Image resizing to 600×600 before processor:** Images are resized before being passed to the ViT processor (which further resizes to 224×224). This preserves more detail from high-resolution leaf images compared to direct 224×224 resizing.
- **Grape dataset exclusion:** The grape-leaf disease dataset was intentionally excluded from the primary training configuration due to suspected data leakage from its uniform white background.
- **HuggingFace Trainer:** Training uses the `Trainer` API with `gradient_accumulation_steps=4` to simulate a larger effective batch size on limited VRAM.
