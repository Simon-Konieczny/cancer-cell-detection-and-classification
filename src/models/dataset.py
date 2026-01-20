import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from pathlib import Path

class ClassificationDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.root = Path(processed_dir)
        df = pd.read_csv(self.root / "metadata.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.split = split
        self.images_dir = self.root / "images"

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            # transforms.ElasticTransform(alpha=50.0, sigma=5.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.RandomErasing(p=0.2 , scale=(0.02, 0.05)),
        ])

        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.label_map = {"benign": 0, "malignant": 1, "normal": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row.img_id

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if self.split == "train":
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)

        label_str = str(row.label).lower().strip()
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        
        return img, label

class SegmentationDataset(Dataset):
    def __init__(self, processed_dir, split="train"):
        self.root = Path(processed_dir)
        df = pd.read_csv(self.root / "metadata.csv")
        self.df = df[df["split"] == split].reset_index(drop=True)

        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.mask_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row.img_id
        mask_path = self.root / row.mask_path

        img = Image.open(img_path).convert("L")
        img = self.img_transform(img)

        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)
        
        return img, mask