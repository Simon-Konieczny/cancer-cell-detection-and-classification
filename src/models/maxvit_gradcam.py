import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import cv2
import os

from class_models import get_model
from dataset import ClassificationDataset as Dataset

def maxvit_reshape_transform(tensor):
    # MaxViT blocks output (B, C, H, W) — already spatial, no reshape needed
    if len(tensor.shape) == 4:
        return tensor  # Already (B, C, H, W)
    # Fallback: if somehow 3D (B, L, C), reshape to spatial
    batch, L, C = tensor.shape
    height = width = int(L**0.5)
    result = tensor.reshape(batch, height, width, C)
    return result.transpose(2, 3).transpose(1, 2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "maxvit_tiny_tf_512"
model = get_model(model_name=model_name)

folder = "pooled_USG"
file_name = "/ablation_architecture:_hybrid_maxvit_fold4.pth"
state_dict = torch.load("./data/processed/" + folder + file_name, map_location=device)
model.load_state_dict(state_dict)
model.to(device).eval()

# --- MaxViT target layer ---
# MaxViT in timm: model.stages[-1].blocks[-1]
# The conv/norm after the attention+MLP in the last block is a good target
target_layers = [model.stages[-1].blocks[-1].conv]  # type: ignore


ds = Dataset("./data/processed/" + folder, indices=[243], split="val")
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=4, shuffle=True)

save_dir = "./gradcam_outputs"
os.makedirs(save_dir, exist_ok=True)

cam = GradCAM(model=model,
              target_layers=target_layers,
              reshape_transform=maxvit_reshape_transform)

for batch_idx, (images, labels) in enumerate(loader):
    images = images.to(device)

    grayscale_cams = cam(input_tensor=images)

    for i in range(images.shape[0]):
        img_tensor = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        grayscale_cam = grayscale_cams[i, :]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        save_path = os.path.join(save_dir, f"{folder}_{file_name[1:].split('.')[0]}_b{batch_idx}_i{i}.png")
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    if batch_idx >= 2:
        break

print("Visualizations saved to:", save_dir)