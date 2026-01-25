import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from mit import get_mit_b3_model, efficient_model
from dataset import SegmentationDataset

def visualize_failures(model, loader, device):
    model.eval()
    failures = []
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            preds = (outputs > 0.5).float().cpu()
            
            for i in range(imgs.size(0)):
                # Calculate Dice for this specific image
                dice = (2 * (preds[i] * masks[i]).sum() + 1e-7) / (preds[i].sum() + masks[i].sum() + 1e-7)
                failures.append((dice.item(), imgs[i].cpu(), masks[i], preds[i]))
    
    # Sort by worst Dice score
    failures.sort(key=lambda x: x[0])
    # sort by best Dice score
    # failures.reverse()
    failures = failures[7:]
    
    # Plot thes
    fig, axes = plt.subplots(5, 3, figsize=(15, 10))
    for i in range(5):
        dice, img, mask, pred = failures[i]
        axes[i, 0].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axes[i, 0].set_title(f"Image (Dice: {dice:.2f})")
        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred.squeeze(), cmap='jet')
        axes[i, 2].set_title("Prediction")
    plt.tight_layout()
    plt.show()

model = get_mit_b3_model()
model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/best_segmentation_model_9.pth"))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
val_ds   = SegmentationDataset("./data/processed/pooled_seg_USG", split="val")
val_loader = DataLoader(val_ds, batch_size=16, shuffle=True)
visualize_failures(model.to(device), val_loader, device)