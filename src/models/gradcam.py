import torch
import torch.nn as nn
import os
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from torch.optim.swa_utils import AveragedModel

from dataset import ClassificationDataset as Dataset
from torch.utils.data import DataLoader
from class_models import get_model
def get_last_layer_name(model):
    """
    Specifically targets the last spatial block in timm models, 
    skipping the non-spatial 'head' and 'norm' layers.
    """
    # Specifically for Swin/MaxViT/EfficientNet
    # We want the last layer that produces a 2D feature map.
    for name, module in reversed(list(model.named_modules())):
        # Skip the final global norm and head
        if "head" in name or "norm" == name.split('.')[-1]:
            continue
            
        # For CNNs: The last Convolution
        if isinstance(module, nn.Conv2d):
            return name
            
        # For ViTs: The last Transformer Block (usually contains the spatial patches)
        # In timm Swin/MaxViT, these are often named 'layers.X.blocks.Y'
        if "blocks" in name and "." in name:
            # We want the block itself, not a sub-layer inside it
            return name
            
    return None

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
    )
    
if device.type == 'cuda':
    torch.cuda.empty_cache()
elif device.type == 'mps':
    torch.mps.empty_cache()

print(f"Using device: {device}")

folders = ["pooled_Mammos", "pooled_USG"]
folder = folders[0]
file_name = "/ablation_vit_swin_tiny_swa_fold4.pth"

model_name = "swin_tiny_patch4_window7_224"


model = get_model(model_name=model_name)
model = AveragedModel(model)
state_dict = torch.load("./data/processed/" + folder + file_name, map_location=device)
model.load_state_dict(state_dict)
model.to(device).eval()

target_layer = get_last_layer_name(model)
print(f"Visualizing focus at layer: {target_layer}")

# ds = Dataset("./data/processed/pooled_Mammos", split="train")
ds = Dataset("./data/processed/" + folder, split="train")
loader = DataLoader(ds, batch_size=4, shuffle=True)

save_dir = "./gradcam_outputs"
os.makedirs(save_dir, exist_ok=True)

with GradCAMpp(model, target_layer=target_layer) as cam_extractor:
    for batch_idx, batch in enumerate(loader):
        images = batch[0].to(device)
        out = model(images)
        
        # 1. Get CAMs
        # Note: some timm models require the output to be passed explicitly 
        class_idxs = out.argmax(dim=-1).tolist()
        multi_layer_cams = cam_extractor(class_idxs, out)
        
        # 2. Extract the actual heatmap tensor
        # Shape is usually [Batch, H, W]
        cams = multi_layer_cams[0]

        for i in range(images.shape[0]):
            print(f"Image {i} - Max Activation: {cams[i].max():.4f}, Min: {cams[i].min():.4f}")
            # 3. Individual Normalization (Crucial for medical images)
            # This forces the "hottest" part of THIS image to be 1.0 (red)
            # and the "coldest" to be 0.0 (blue)
            cam_img = cams[i]
            cam_min, cam_max = cam_img.min(), cam_img.max()
            
            # Prevent division by zero if the heatmap is totally flat
            if cam_max > cam_min:
                cam_img = (cam_img - cam_min) / (cam_max - cam_min)
            else:
                cam_img = torch.zeros_like(cam_img)

            # 4. Prepare Background (Denormalize)
            img_tensor = images[i].cpu()
            # Standard ImageNet denormalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_pil = to_pil_image(img_tensor.clamp(0, 1))

            # 5. Overlay
            # Convert heatmap to PIL 'F' mode (float32)
            mask_pil = to_pil_image(cam_img.cpu(), mode='F')
            
            # alpha=0.5: 0 is original image, 1 is heatmap. 
            # If it's still too blue, lower alpha to 0.3 to see the tissue better
            result = overlay_mask(img_pil, mask_pil, colormap="jet", alpha=0.5)
            
            # 6. Save
            save_path = os.path.join(save_dir, f"{folder}_{file_name[1:].split('.')[0]}_b{batch_idx}_i{i}.png")
            result.save(save_path)
        
        # Dissertation Tip: Only generate a few samples to check quality first!
        if batch_idx >= 2: 
            break