import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

class BreastDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.root = Path(processed_dir)
        df = pd.read_csv(self.root / "metadata.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.images_dir = self.root / "images"

        self.transform = transforms.Compose([
            transforms.ToTensor(),               # HWC -> CHW, scales to [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # simple baseline
        ])

        self.label_map = {
            "Benign": 0,
            "Malignant": 1,
            "Normal": 2
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row.img_id

        img = Image.open(img_path).convert("L")   # using grayscale for USG + MG
        img = self.transform(img)

        label = torch.tensor(self.label_map[row.label], dtype=torch.long)
        return img, label