import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import cv2
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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

        if split == "train":
            # consider adding more augmentations here like elastic transform
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                # Mimics USG probe pressure
                A.ElasticTransform(alpha=0.5, sigma=25, p=0.2),
                # Mimics USG speckle noise
                A.GaussNoise(var_limit=(10, 50), p=0.3), 
                A.RandomBrightnessContrast(p=0.3),
                # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # self.transform = A.Compose([
            #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #     ToTensorV2(),
            # ])
            self.transform = A.Compose([
                # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.root / "images" / row.img_id))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.root / row.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
                mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
            
        if mask.ndimension() == 2:
            mask = mask.unsqueeze(0)
        
        return img, mask