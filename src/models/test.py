import matplotlib.pyplot as plt
import torchvision
from dataset import ClassificationDataset as ClassificationDataset
from torch.utils.data import DataLoader

def visualize_batch(dataloader):
    batch_imgs, batch_labels = next(iter(dataloader))
    
    grid = torchvision.utils.make_grid(batch_imgs, nrow=4)
    grid = grid * 0.5 + 0.5  # Unnormalize
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title("Visualizing Augmented Training Batch")
    plt.axis('off')
    plt.show()

# Usage:
train_loader = DataLoader(ClassificationDataset("./data/processed/BrCaWisconsin", split="train"), batch_size=16, shuffle=True)
visualize_batch(train_loader)