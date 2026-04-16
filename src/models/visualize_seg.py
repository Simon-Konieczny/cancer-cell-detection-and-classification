import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
import os
import pandas as pd
import random
import math

from dataset import SegmentationDataset as DataSet
from mit import get_model_resnet, get_mit_b3_model, get_model_resnet_unet

def visualize_ablation_results(models_dict, loader, device, save_dir, max_cols=7):
    # Convert loader to list for easy indexing by sample group
    samples = list(loader)
    num_samples = len(samples)
    model_items = list(models_dict.items())
    num_models = len(model_items)
    
    samples_per_block = 3
    usable_cols = 7
    
    # Calculate how many "sets of rows" we need per sample group to show all models
    # First row-set shows 1 model (alongside Input/GT). 
    # Remaining models are shown 3 at a time in subsequent row-sets.
    extra_models = max(0, num_models - 1)
    model_blocks_per_sample_group = 1 + math.ceil(extra_models / usable_cols)
    
    num_sample_groups = math.ceil(num_samples / samples_per_block)
    total_rows = num_sample_groups * model_blocks_per_sample_group * samples_per_block

    fig, axes = plt.subplots(total_rows, max_cols, figsize=(15, total_rows))
    
    # Flatten models for iteration
    model_names = [m[0] for m in model_items]
    model_objs = [m[1] for m in model_items]

    for s_grp_idx in range(num_sample_groups):
        for m_blk_idx in range(model_blocks_per_sample_group):
            # Iterate through the 3 samples in this block
            for s_offset in range(samples_per_block):
                sample_idx = s_grp_idx * samples_per_block + s_offset
                
                # Calculate the exact row in the big grid
                # (Sample group offset) + (Model wrap offset) + (Position within the 3 rows)
                row_idx = (s_grp_idx * model_blocks_per_sample_group * samples_per_block) + \
                          (m_blk_idx * samples_per_block) + s_offset
                
                # Safety: if we run out of samples (e.g., total samples is 10, not 12)
                if sample_idx >= num_samples:
                    for c in range(max_cols):
                        axes[row_idx, c].axis('off')
                    continue

                img, mask = samples[sample_idx]
                input_tensor = img.to(device)
                
                # Prepare Image for plotting
                image_np = img.squeeze().cpu().numpy()
                if image_np.ndim == 3: image_np = np.transpose(image_np, (1, 2, 0))
                image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
                mask_np = mask.squeeze().cpu().numpy()

                # Case A: First block for these samples (Input + GT + 1st Model)
                if m_blk_idx == 0:
                    # Col 0: Input
                    axes[row_idx, 0].imshow(image_np, cmap='gray')
                    axes[row_idx, 0].set_ylabel(f"S{sample_idx} | Input", fontsize=9)
                    
                    # Col 1: GT
                    axes[row_idx, 1].imshow(image_np, cmap='gray')
                    axes[row_idx, 1].imshow(mask_np, cmap='spring', alpha=0.4)
                    axes[row_idx, 1].set_title("GT", fontsize=9)
                    
                    # Col 2: First Model (if exists)
                    if num_models > 0:
                        _plot_model_on_ax(axes[row_idx, 2], model_objs[0], model_names[0], 
                                          input_tensor, image_np, device, sample_idx)
                        _plot_model_on_ax(axes[row_idx, 3], model_objs[1], model_names[1], 
                                          input_tensor, image_np, device, sample_idx)
                        _plot_model_on_ax(axes[row_idx, 4], model_objs[2], model_names[2], 
                                          input_tensor, image_np, device, sample_idx)
                        _plot_model_on_ax(axes[row_idx, 5], model_objs[3], model_names[3], 
                                          input_tensor, image_np, device, sample_idx)
                        _plot_model_on_ax(axes[row_idx, 6], model_objs[4], model_names[4], 
                                          input_tensor, image_np, device, sample_idx)
                
                # Case B: Subsequent blocks (Filling 3 models per row)
                else:
                    # Logic to find which models to show: 
                    # 1st model was used in block 0. 
                    # Block 1 starts at model index 1, Block 2 at index 4, etc.
                    model_start_idx = 1 + (m_blk_idx - 1) * usable_cols
                    
                    for col_offset in range(usable_cols):
                        current_model_idx = model_start_idx + col_offset + 4
                        ax = axes[row_idx, col_offset]
                        
                        if current_model_idx < num_models:
                            _plot_model_on_ax(ax, model_objs[current_model_idx], 
                                              model_names[current_model_idx], 
                                              input_tensor, image_np, device, sample_idx)
                        else:
                            ax.axis('off')

                # Strictly hide the last 2 columns as requested
                # axes[row_idx, 3].axis('off')
                # axes[row_idx, 4].axis('off')
                
                # Hide unused axes in the active row (if m_blk_idx == 0 and num_models == 0)
                # or if we are in Case A but only have Input/GT.
                for c in range(usable_cols):
                    if not axes[row_idx, c].images: # If nothing was plotted
                        axes[row_idx, c].axis('off')

    plt.tight_layout()
    # Add a visual gap between sample groups
    plt.subplots_adjust(hspace=0.6)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "segmentation_comparison_small.png"), bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

def _plot_model_on_ax(ax, model, name, input_tensor, image_np, device, sample_idx):
    model.to(device).eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
    ax.imshow(image_np, cmap='gray')
    ax.imshow(pred, cmap='summer', alpha=0.5)
    ax.set_title(name, fontsize=8)
    ax.set_ylabel(f"S{sample_idx} | Input", fontsize=9)
    ax.axis('off')

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
                    hd95_metric(y_pred=preds.cpu(), y=labels.cpu())

            final_hd95 = hd95_metric.aggregate().item() #type: ignore
            hd95_fold.append(final_hd95)
            print(f"Mean HD95 for Fold {r + 1}: {final_hd95}")
        
        results.append({
            "Experiment": exp_name,
            "HD95s": hd95_fold
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "hd95_results_new.csv"), index=False)
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

models_to_compare = {
    # "ResNet Adam Simple Baseline": load_model(get_model_resnet(), "ablation_resnet_adam_simple_baseline_fold2.pth", device),
    # "ResNet Adam Simple 0.3 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_simple_0.3_bce_fold2.pth", device),
    # "Resnet AdamW Simple Tversky": load_model(get_model_resnet(), "ablation_resnet_adamw_simple_tversky_fold2.pth", device),
    # "ResNet Adam Simple 0.2 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_simple_0.2_bce_fold4.pth", device),
    # "ResNet Adam Simple Focal": load_model(get_model_resnet(), "ablation_resnet_adam_simple_focal_fold2.pth", device),

    # "Resnet Adam Baseline": load_model(get_model_resnet(), "ablation_resnet_adam_baseline_fold2.pth", device),
    # "Resnet Adam 0.3 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_0.3_bce_fold5.pth", device),
    "Resnet Adam 0.2 BCE": load_model(get_model_resnet(), "ablation_resnet_adam_0.2_bce_fold2.pth", device),
    # "Resnet Tversky": load_model(get_model_resnet(), "ablation_resnet_tversky_fold4.pth", device),
    # "ResNet Adam Focal": load_model(get_model_resnet(), "ablation_resnet_adam_focal_fold5.pth", device),

    # "Resnet 224": load_model(get_model_resnet(), "ablation_resnet_224_fold4.pth", device),
    # "Resnet 1024": load_model(get_model_resnet(), "ablation_resnet_1024_fold2.pth", device),
    "Resnet Aggressive": load_model(get_model_resnet(), "ablation_resnet_aggressive_fold4.pth", device),
    # "Resnet Unet": load_model(get_model_resnet_unet(), "ablation_resnet_unet_fold4.pth", device),

    # "MIT-B3 Adam Simple Baseline": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_simple_baseline_fold2.pth", device),
    # "MIT-B3 Adam Simple Tversky": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_simple_tversky_fold4.pth", device),
    # "MIT-B3 Adam Simple 0.2 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_simple_0.2_bce_fold2.pth", device),
    "MIT-B3 Adam Simple Focal": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_simple_focal_fold2.pth", device),
    # "MIT-B3 Adam Simple 0.3 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_simple_0.3_bce_fold5.pth", device),
    
    # "MIT-B3 Adam": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_fold3.pth", device),
    # "MIT-B3 Adam Tversky": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_tversky_fold5.pth", device),
    # "MIT-B3 Adam 0.3 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_0.3_bce_fold2.pth", device),
    # "MIT-B3 Adam Focal": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_focal_fold4.pth", device),
    # "MIT-B3 Adam 0.2 BCE": load_model(get_mit_b3_model(), "ablation_mit-b3_adam_0.2_bce_fold4.pth", device),
    
    # "MIT-B3 Aggressive": load_model(get_mit_b3_model(), "ablation_mit-b3_aggressive_fold2.pth", device),
    # "MIT-B3 224": load_model(get_mit_b3_model(), "ablation_mit-b3_224_fold2.pth", device),
}

folds = {
    "Resnet No Augmentation": (get_model_resnet(), "ablation_no_augmentation_baseline_fold"),
    "ResNet Adam Simple Baseline": (get_model_resnet(), "ablation_resnet_adam_simple_baseline_fold"),
    "ResNet Adam Simple 0.3 BCE": (get_model_resnet(), "ablation_resnet_adam_simple_0.3_bce_fold"),
    "Resnet AdamW Simple Tversky": (get_model_resnet(), "ablation_resnet_adamw_simple_tversky_fold"),
    "ResNet Adam Simple 0.2 BCE": (get_model_resnet(), "ablation_resnet_adam_simple_0.2_bce_fold"),
    "ResNet Adam Simple Focal": (get_model_resnet(), "ablation_resnet_adam_simple_focal_fold"),

    "Resnet Adam Baseline": (get_model_resnet(), "ablation_resnet_adam_baseline_fold"),
    "ResNet Adam 0.3 BCE": (get_model_resnet(), "ablation_resnet_adam_0.3_bce_fold"),
    "ResNet Adam 0.2 BCE": (get_model_resnet(), "ablation_resnet_adam_0.2_bce_fold"),
    "Resnet Tversky": (get_model_resnet(), "ablation_resnet_tversky_fold"),
    "ResNet Adam Focal": (get_model_resnet(), "ablation_resnet_adam_focal_fold"),

    "Resnet 224": (get_model_resnet(), "ablation_resnet_224_fold"),
    "Resnet 1024": (get_model_resnet(), "ablation_resnet_1024_fold"),
    "Resnet Aggressive": (get_model_resnet(), "ablation_resnet_aggressive_fold"),
    "Resnet Unet": (get_model_resnet_unet(), "ablation_resnet_unet_fold"),
    # "Resnet Deep": (resnet_deeplabv3, []),

    "MIT-B3 Adam Simple Baseline": (get_mit_b3_model(), "ablation_mit-b3_adam_simple_baseline_fold"),
    "MIT-B3 Adam Simple Tversky": (get_mit_b3_model(), "ablation_mit-b3_adam_simple_tversky_fold"),
    "MIT-B3 Adam Simple 0.2 BCE": (get_mit_b3_model(), "ablation_mit-b3_adam_simple_0.2_bce_fold"),
    "MIT-B3 Adam Simple Focal": (get_mit_b3_model(), "ablation_mit-b3_adam_simple_focal_fold"),
    "MIT-B3 Adam Simple 0.3 BCE": (get_mit_b3_model(), "ablation_mit-b3_adam_simple_0.3_bce_fold"),

    "MIT-B3 Adam": (get_mit_b3_model(), "ablation_mit-b3_adam_fold"),
    "MIT-B3 Adam Tversky": (get_mit_b3_model(), "ablation_mit-b3_adam_tversky_fold"),
    "MIT-B3 Adam 0.3 BCE": (get_mit_b3_model(), "ablation_mit-b3_adam_0.3_bce_fold"),
    "MIT-B3 Adam Focal": (get_mit_b3_model(), "ablation_mit-b3_adam_focal_fold"),
    "MIT-B3 Adam 0.2 BCE": (get_mit_b3_model(), "ablation_mit-b3_adam_0.2_bce_fold"),

    "MIT-B3 Aggressive": (get_mit_b3_model(), "ablation_mit-b3_aggressive_fold"),
    "MIT-B3 224": (get_mit_b3_model(), "ablation_mit-b3_224_fold"),
    # "MIT-B3 1024": (mit_b3, []),
    # "MIT-B3 PlusPlus Tversky": (mit_b3_unet_plus_plus, []),
    # "MIT-B3 Deep": (mit_b3_deeplabv3, []),
}

random_indeces = random.sample(range(1,101), 8)

val_ds = DataSet(processed_dir, split="val", indices=random_indeces)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

calculateHausdorff95(folds, val_loader, device=device, save_dir=seg_metrics_save_dir)

# val_ds = DataSet(processed_dir, split="val", indices=indices_to_test)
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
# visualize_ablation_results(models_to_compare, val_loader, device=device, save_dir=vis_save_dir)