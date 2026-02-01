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

def get_bbox(img):
    """
    Finds the bounding box of the breast tissue to remove empty black background.
    """
    # 1. Threshold to create a mask of the tissue
    # Mammograms have very dark backgrounds, so a low threshold works
    _, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    
    # 2. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, img.shape[1], img.shape[0]
        
    # 3. Get the largest contour (the breast)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

class ClassificationDataset(Dataset):
    def __init__(self, processed_dir, split="train", indices=None, use_roi=True):
        self.root = Path(processed_dir)
        self.use_roi = use_roi
        df = pd.read_csv(self.root / "metadata.csv")
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
        else:
            self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = self.root / "images"

        # ImageNet constants
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if split == "train":
            self.transform = A.Compose([
                    A.LongestMaxSize(max_size=640),
                    A.PadIfNeeded(
                        min_height=640, 
                        min_width=640, 
                        border_mode=cv2.BORDER_CONSTANT, 
                        value=0
                    ),
                    A.Resize(height=640, width=640),
                    A.OneOf([
                        A.Sharpen(alpha=(0.2, 0.5), p=1.0),
                        A.CLAHE(clip_limit=4.0, p=1.0), 
                    ], p=0.5),
                    A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
                    # Mimics USG probe pressure
                    # A.ElasticTransform(alpha=0.5, sigma=25, p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                    A.OneOf([
                        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, p=0.5),
                        A.GridDropout(ratio=0.2, p=1.0),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
                    ], p=0.3),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ])
        else:
            self.transform = A.Compose([
                    A.LongestMaxSize(max_size=640),
                    A.PadIfNeeded(
                        min_height=640, 
                        min_width=640, 
                        border_mode=cv2.BORDER_CONSTANT, 
                        value=0
                    ),
                    A.Resize(height=640, width=640),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ])

        self.label_map = {"benign": 0, "malignant": 1, "normal": 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row.img_id

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if self.use_roi:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x, y, w, h = get_bbox(gray)

            if w > 10 and h > 10:
                img = img[y:y+h, x:x+w]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image = img)
        img = augmented['image']

        label_str = str(row.label).lower().strip()
        label = self.label_map[label_str]
        
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