import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
import os
import pandas as pd

from dataset import SegmentationDataset as DataSet
from mit import get_model_resnet, get_mit_b3_model, get_model_resnet_deeplabv3, get_model_resnet_unet, get_mit_b3_deeplabv3_model

def visualize_ablation_results(models_dict, loader, device, save_dir):
    num_samples = len(loader)
    num_models = len(models_dict)
    
    # Grid: Input, GT, then one column per model
    fig, axes = plt.subplots(num_samples, num_models + 2, figsize=((num_models) * 4, num_samples * 2))
    
    if num_samples == 1:
        axes = np.atleast_2d(axes)

    for r, (img, mask) in enumerate(loader):
        input_tensor = img.to(device)
        
        # Prepare background image (H, W, C)
        image_np = img.squeeze().cpu().numpy()
        if image_np.ndim == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Normalize image to [0, 1] for consistent plotting if it isn't already
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        mask_np = mask.squeeze().cpu().numpy()

        # 1. Plot Original Image
        axes[r, 0].imshow(image_np, cmap='gray')
        axes[r, 0].set_title(f"Sample {r} | Input")
        axes[r, 0].axis('off')

        # 2. Plot Ground Truth Overlay
        axes[r, 1].imshow(image_np, cmap='gray')
        axes[r, 1].imshow(mask_np, cmap='spring', alpha=0.4) # Pinkish overlay for GT
        axes[r, 1].set_title("Ground Truth")
        axes[r, 1].axis('off')

        # 3. Plot Each Model Prediction Overlay
        for c, (name, model) in enumerate(models_dict.items()):
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_bin = (pred > 0.5).astype(np.float32)
            
            # Draw the base image
            axes[r, c + 2].imshow(image_np, cmap='gray')
            # Draw the prediction on top (Green/Yellow 'summer' or 'jet')
            # Using alpha=0.5 makes the underlying anatomy visible
            axes[r, c + 2].imshow(pred_bin, cmap='summer', alpha=0.5)
            axes[r, c + 2].set_title(name)
            axes[r, c + 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "segmentation_comparison.png"))
    plt.show()
    plt.close()
    print(f"Segmentation comparison saved to {os.path.join(save_dir, 'segmentation_comparison.png')}")

def calculateHausdorff95(folds_dict, val_loader, device, save_dir):
    results = []
    # Initialize the metric (95th percentile is standard)
    hd95_metric = HausdorffDistanceMetric(percentile=95, reduction="mean")

    for exp_name, (model, base_path) in folds_dict.items():
        fold_paths = []
        for i in range(1,6):
            fold_paths.append(base_path + str(i) + ".pth")

        hd95_fold = []
        for r, fold_path in enumerate(fold_paths):
            hd95_metric.reset()
            model = load_model(model, fold_path, device)
            model.to(device)
            model.eval()

            with torch.no_grad():
                for batch in val_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = (outputs > 0.5).float() # Binarize
                    
                    # Calculate HD95
                    hd95_metric(y_pred=preds, y=labels)

            final_hd95 = hd95_metric.aggregate()
            hd95_fold.append(final_hd95)
            print(f"Mean HD95 for Fold {r + 1}: {final_hd95}")
        
        results.append({
            "Experiment": exp_name,
            "HD95s": hd95_fold
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "hd95_results.csv"), index=False)
    print(f"HD95 results saved to {os.path.join(save_dir, 'hd95_results.csv')}")

def load_model(model, file_path, device):
    model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/" + file_path, map_location=device))
    return model

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

vis_save_dir = "./segmentation_visuals"
seg_metrics_save_dir = "./segmentation_metrics"
os.makedirs(vis_save_dir, exist_ok=True)
os.makedirs(seg_metrics_save_dir, exist_ok=True)

processed_dir="./data/processed/pooled_seg_USG"
indices_to_test = [10, 55, 120]

resnet = get_model_resnet()
resnet_unet = get_model_resnet_unet()
resnet_deeplabv3 = get_model_resnet_deeplabv3()

mit_b3 = get_mit_b3_model()
mit_b3_deeplabv3 = get_mit_b3_deeplabv3_model()

models_to_compare = {
    "ResNet Adam Simple Baseline": load_model(get_model_resnet(), "ablation_resnet_adam_simple_baseline_fold2.pth", device),
    "ResNet Adam Simple 0.3 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_simple_0.3_bce_fold2.pth", device),
    "Resnet AdamW Simple Tversky": load_model(get_model_resnet(), "ablation_resnet_adamw_simple_tversky_fold2.pth", device),
    "ResNet Adam Simple 0.2 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_simple_0.2_bce_fold4.pth", device),
    "ResNet Adam Simple Focal": load_model(get_model_resnet(), "ablation_resnet_adam_simple_focal_fold2.pth", device),

    "Resnet Adam Baseline": load_model(get_model_resnet(), "ablation_resnet_adam_baseline_fold2.pth", device),
    # "Resnet Adam 0.3 BCE": load_model(resnet, ".pth", device),
    # "Resnet Adam 0.2 BCE": load_model(resnet, ".pth", device),
    # "Resnet Tversky": load_model(resnet, ".pth", device),
    # "ResNet Adam Focal": load_model(resnet, ".pth", device),

    "Resnet 224": load_model(get_model_resnet(), "ablation_resnet_224_fold4.pth", device),
    "Resnet 1024": load_model(get_model_resnet(), "ablation_resnet_1024_fold2.pth", device),
    "Resnet Aggressive": load_model(get_model_resnet(), "ablation_resnet_aggressive_fold4.pth", device),
    "Resnet Unet": load_model(get_model_resnet_unet(), "ablation_resnet_unet_fold4.pth", device),
    # "Resnet Deep": load_model(resnet_deeplabv3, ".pth", device),

    "MIT-B3 Adam Simple Baseline": load_model(mit_b3, "ablation_mit-b3_adam_simple_baseline_fold2.pth", device),
    # "MIT-B3 Adam Simple Tversky": load_model(mit_b3, ".pth", device),
    # "MIT-B3 Adam Simple 0.2 BCE": load_model(mit_b3, ".pth", device),
    # "MIT-B3 Adam Simple Focal": load_model(mit_b3, ".pth", device),
    # "MIT-B3 Adam Simple 0.3 BCE": load_model(mit_b3, ".pth", device),
    
    "MIT-B3 Adam": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_fold3.pth", device),
    "MIT-B3 Adam Tversky": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_tversky_fold5.pth", device),
    "MIT-B3 Adam 0.3 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_0.3_bce_fold2.pth", device),
    "MIT-B3 Adam Focal": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_focal_fold4.pth", device),
    "MIT-B3 Adam 0.2 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_0.2_bce_fold4.pth", device),
    
    "MIT-B3 Aggressive": load_model(get_mit_b3_model(), "ablation_mit-b3_aggressive_fold2.pth", device),
    "MIT-B3 224": load_model(get_mit_b3_model(), "ablation_mit-b3_224_fold2.pth", device),
    # "MIT-B3 1024": load_model(mit_b3, ".pth", device),
    # "MIT-B3 PlusPlus Tversky": load_model(mit_b3_unet_plus_plus, ".pth"),
    # "MIT-B3 Deep": load_model(mit_b3_deeplabv3, ".pth", device),
}

folds = {
    # "ResNet Adam Simple Baseline": (resnet, []),
    # "ResNet Adam Simple 0.3 BCE": (resnet, []),
    # "Resnet AdamW Simple Tversky": (resnet, []),
    # "ResNet Adam Simple 0.2 BCE": (resnet, []),
    # "ResNet Adam Simple Focal": (resnet, []),

    # "Resnet Adam Baseline": (resnet, []),
    # "Resnet Adam 0.3 BCE": (resnet, []),
    # "Resnet Adam 0.2 BCE": (resnet, []),
    # "Resnet Tversky": (resnet, []),
    # "ResNet Adam Focal": (resnet, []),

    # "Resnet 224": (resnet, []),
    # "Resnet 1024": (resnet, []),
    # "Resnet Aggressive": (resnet, []),
    # "Resnet Unet": (resnet_unet, []),
    # "Resnet Deep": (resnet_deeplabv3, []),

    # "MIT-B3 Adam Simple Baseline": (mit_b3, []),
    # "MIT-B3 Adam Simple Tversky": (mit_b3, []),
    # "MIT-B3 Adam Simple 0.2 BCE": (mit_b3, []),
    # "MIT-B3 Adam Simple Focal": (mit_b3, []),
    # "MIT-B3 Adam Simple 0.3 BCE": (mit_b3, []),

    "MIT-B3 Adam": (mit_b3, "ablation_mit-b3_adam_fold"),
    "MIT-B3 Adam Tversky": (mit_b3, "ablation_mit-b3_adam_tversky_fold"),
    "MIT-B3 Adam 0.3 BCE": (mit_b3, "ablation_mit-b3_adam_0.3_bce_fold"),
    "MIT-B3 Adam Focal": (mit_b3, "ablation_mit-b3_adam_focal_fold"),
    "MIT-B3 Adam 0.2 BCE": (mit_b3, "ablation_mit-b3_adam_0.2_bce_fold"),

    "MIT-B3 Aggressive": (mit_b3, "ablation_mit-b3_aggressive_fold"),
    "MIT-B3 224": (mit_b3, "ablation_mit-b3_224_fold"),
    # "MIT-B3 1024": (mit_b3, []),
    # "MIT-B3 PlusPlus Tversky": (mit_b3_unet_plus_plus, []),
    # "MIT-B3 Deep": (mit_b3_deeplabv3, []),
}

val_ds = DataSet(processed_dir, split="val", indices=indices_to_test)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# calculateHausdorff95(folds, val_loader, device=device, save_dir=seg_metrics_save_dir)

visualize_ablation_results(models_to_compare, val_loader, device=device, save_dir=vis_save_dir)