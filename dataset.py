import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

RESIZE_DIM = 600

class Leaves(Dataset):
    def __init__(self, processor, csv_path: str, split: str):
        assert split in ("train", "valid", "test"), f"Invalid split: {split}"

        self.processor = processor
        self.split = split
        self.root = os.path.join(os.path.dirname(csv_path), split)

        split_files = set(os.listdir(self.root))
        df = pd.read_csv(csv_path)
        df = df[df["filename"].isin(split_files)].reset_index(drop=True)
        self.df = df

        self.name2idx = {"healthy": 0, "mildew": 1}
        self.idx2name = {0: "healthy", 1: "mildew"}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["filename"])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (RESIZE_DIM, RESIZE_DIM))

        encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        label = self.name2idx["healthy"] if row["healthy"] == 1 else self.name2idx["mildew"]

        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}


def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}
