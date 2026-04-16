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
    for name, module in reversed(list(model.named_modules())):
        if "head" in name or "norm" == name.split('.')[-1]:
            continue
            
        if isinstance(module, nn.Conv2d):
            return name
            
        if "blocks" in name and "." in name:
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

folders = ["pooled_Mammos/", "pooled_USG/"]
folder = folders[1]
file_name = "ablation_architecture:_efficientnetv2-s_fold5.pth"

model_name = "convnext_small"


model = get_model(model_name=model_name)
model = AveragedModel(model)
state_dict = torch.load("./data/processed/" + folder + file_name, map_location=device)
model.load_state_dict(state_dict)
model.to(device).eval()

target_layer = get_last_layer_name(model)
print(f"Visualizing focus at layer: {target_layer}")

ds = Dataset("./data/processed/" + folder, indices=[243], split="val")
loader = DataLoader(ds, batch_size=4, shuffle=True)

save_dir = "./gradcam_outputs/" + folder
os.makedirs(save_dir, exist_ok=True)

with GradCAMpp(model, target_layer=target_layer) as cam_extractor:
    for batch_idx, batch in enumerate(loader):
        images = batch[0].to(device)
        out = model(images)
        
        class_idxs = out.argmax(dim=-1).tolist()
        multi_layer_cams = cam_extractor(class_idxs, out)
        
        cams = multi_layer_cams[0]

        for i in range(images.shape[0]):
            print(f"Image {i} - Max Activation: {cams[i].max():.4f}, Min: {cams[i].min():.4f}")
            cam_img = cams[i]
            cam_min, cam_max = cam_img.min(), cam_img.max()
            
            if cam_max > cam_min:
                cam_img = (cam_img - cam_min) / (cam_max - cam_min)
            else:
                cam_img = torch.zeros_like(cam_img)

            img_tensor = images[i].cpu()
            # Standard ImageNet denormalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = img_tensor * std + mean
            img_pil = to_pil_image(img_tensor.clamp(0, 1))

            mask_pil = to_pil_image(cam_img.cpu(), mode='F')
            
            result = overlay_mask(img_pil, mask_pil, colormap="jet", alpha=0.5)
            
            save_path = os.path.join(save_dir, f"{file_name.split('.')[0]}_b{batch_idx}_i{i}.png")
            result.save(save_path)
        
        if batch_idx >= 2: 
            break