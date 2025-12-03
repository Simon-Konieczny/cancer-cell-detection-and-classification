import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

class ClassificationDataset(Dataset):
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
    
class SegmentationDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.root = Path(processed_dir)
        df = pd.read_csv(self.root / "metadata.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),               # HWC -> CHW, scales to [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # simple baseline
        ])

        self.mask_transform = transforms.ToTensor()  # masks will be converted to [0,1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row.img_id
        mask_path = self.masks_dir / row.mask_id

        img = Image.open(img_path).convert("L")   # using grayscale for USG + MG
        img = self.img_transform(img)

        mask = Image.open(mask_path).convert("L")  # single channel mask
        mask = self.mask_transform(mask)           # [1,H,W], values in [0,1]
        
        return img, mask