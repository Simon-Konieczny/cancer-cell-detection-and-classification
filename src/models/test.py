import matplotlib.pyplot as plt
import torchvision
from dataset import SegmentationDataset as Dataset
from dataset import ClassificationDataset as ClsDataset
from torch.utils.data import DataLoader
import numpy as np

def visualize_batch(dataloader):
    batch_imgs, batch_labels = next(iter(dataloader))
    
    grid = torchvision.utils.make_grid(batch_imgs, nrow=4)
    grid = grid * 0.5 + 0.5  # Unnormalize
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title("Visualizing Augmented Training Batch")
    plt.axis('off')
    plt.show()

# Classification
train_loader = DataLoader(ClsDataset("./data/processed/pooled_USG", split="test"), 
                          batch_size=16,)
visualize_batch(train_loader)

def visualize_sample(dataset, idx=0):
    img, mask = dataset[idx]
    
    # Un-normalize the image for display
    # (Assuming mean=0.5, std=0.5. Adjust if using ImageNet stats)
    img_display = img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    img_display = np.clip(img_display, 0, 1)
    
    mask_display = mask.squeeze().numpy()

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Original USG Image")
    plt.imshow(img_display)
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_display, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Overlay (Alignment Check)")
    plt.imshow(img_display)
    # Overlay mask with 30% transparency
    plt.imshow(mask_display, alpha=0.3, cmap='Reds') 
    
    plt.show()

#Segmentation
# visualize_sample(Dataset("./data/processed/pooled_seg_USG", split="train"), idx=17)