import torch
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import cv2

from dataset import SegmentationDataset as Dataset
from mit import get_model_resnet, get_mit_b3_model, get_model_resnet_deeplabv3, get_model_resnet_unet, get_mit_b3_deeplabv3_model
from losses import HybridLoss, HybridFocalDiceLoss, HybridLoss2
from train_segmentation import _train_with_stop, validate
from train_classification import _save_history

def run_ablation_study(processed_dir, logger, batch_size, k_folds, epochs):
    root = Path(processed_dir)
    results = []
    full_df = pd.read_csv(root / "metadata.csv")
    mask_paths = full_df["mask_path"].tolist()
    areas = []

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    elif device.type == 'mps':
        torch.mps.empty_cache()
    print(f"Using device: {device}")

    for p in mask_paths:
        mask = cv2.imread(processed_dir + '/' + p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask at {p}. Skipping area calculation for this sample.")
            continue
        area_ratio = np.sum(mask > 0) / mask.size 
        areas.append(area_ratio)
    full_df["area"] = areas
    full_df['stratify_label'] = pd.qcut(full_df['area'], q=k_folds, labels=False, duplicates='drop')

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    auto_weight = calculate_dataset_weight(full_df, processed_dir)
    if (device.type == 'mps'):
        auto_weight = auto_weight.astype(np.float32) # type: ignore
    weight_tensor = torch.tensor([auto_weight]).to(device)
    print(f"Calculated Ratio Weight: {auto_weight:.2f}")

    experiments = [
        # NO AUG BASELINE
        {"name": "No Augmentation Baseline", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "bce_with_logits", "any_augmentation": False},
        
        #ResNet34
        # {"name": "ResNet Adam Simple Baseline", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "bce_with_logits"},
        # {"name": "ResNet Adam Simple 0.3 BCE", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "hybrid_loss_0.3"},
        # {"name": "Resnet AdamW Simple Tversky", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "tversky"},
        # {"name": "ResNet Adam Simple 0.2 BCE", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "hybrid_loss_0.2"},
        # {"name": "ResNet Adam Simple Focal", "encoder": "resnet34", "optimizer": "adamw_simple", "criterion": "focal_dice"},

        # {"name": "Resnet Adam Baseline", "encoder": "resnet34", "optimizer": "adamw", "criterion": "bce_with_logits"},
        # {"name": "ResNet Adam 0.3 BCE", "encoder": "resnet34", "optimizer": "adamw", "criterion": "hybrid_loss_0.3"},
        # {"name": "ResNet Adam 0.2 BCE", "encoder": "resnet34", "optimizer": "adamw", "criterion": "hybrid_loss_0.2"},
        # {"name": "Resnet Tversky", "encoder": "resnet34", "optimizer": "adamw", "criterion": "tversky"},
        # {"name": "ResNet Adam Focal", "encoder": "resnet34", "optimizer": "adamw", "criterion": "focal_dice"},

        # {"name": "Resnet 224", "encoder": "resnet34", "optimizer": "adamw", "criterion": "tversky", "size": 224},
        # {"name": "Resnet 1024", "encoder": "resnet34", "optimizer": "adamw", "criterion": "tversky", "size": 1024},

        # {"name": "Resnet Aggressive", "encoder": "resnet34", "optimizer": "adamw", "criterion": "tversky", "aggressive": True},

        # {"name": "Resnet Unet", "encoder": "resnet_unet", "optimizer": "adamw", "criterion": "tversky"},

        # {"name": "Resnet Deep", "encoder": "resnet_deeplabv3", "optimizer": "adamw", "criterion": "tversky"}, # errored out expected more that 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])

        # MiT
        # {"name": "MIT-B3 Adam Simple Baseline", "encoder": "mit_b3", "optimizer": "adamw_simple", "criterion": "bce_with_logits"},
        # {"name": "MIT-B3 Adam Simple Tversky", "encoder": "mit_b3", "optimizer": "adamw_simple", "criterion": "tversky"},
        # {"name": "MIT-B3 Adam Simple 0.2 BCE", "encoder": "mit_b3", "optimizer": "adamw_simple", "criterion": "hybrid_loss_0.2"},
        # {"name": "MIT-B3 Adam Simple Focal", "encoder": "mit_b3", "optimizer": "adamw_simple", "criterion": "focal_dice"},
        # {"name": "MIT-B3 Adam Simple 0.3 BCE", "encoder": "mit_b3", "optimizer": "adamw_simple", "criterion": "hybrid_loss_0.3"},
        
        # {"name": "MIT-B3 Adam", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "bce_with_logits"},
        # {"name": "MIT-B3 Adam Tversky", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "tversky"},
        # {"name": "MIT-B3 Adam 0.3 BCE", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "hybrid_loss_0.3"},
        # {"name": "MIT-B3 Adam Focal", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "focal_dice"},
        # {"name": "MIT-B3 Adam 0.2 BCE", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "hybrid_loss_0.2"},

        # {"name": "MIT-B3 Aggressive", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "tversky", "aggressive": True},

        # {"name": "MIT-B3 224", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "tversky", "size": 224},
        #{"name": "MIT-B3 1024 Tversky", "encoder": "mit_b3", "optimizer": "adamw", "criterion": "tversky", "size": 1024}, # OOM

        # {"name": "MIT-B3 PlusPlus Tversky", "encoder": "mit_b3_unet_plus_plus", "optimizer": "adamw", "criterion": "tversky"}, # ERROR: UnetPlusPlus not supported

        #{"name": "MIT-B3 DeeplabV3", "encoder": "mit_b3_deeplabv3", "optimizer": "adamw", "criterion": "tversky"}, # OOM

    ]

    for exp in experiments:
        try:
            final_model_path = Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{k_folds}.pth"

            if final_model_path.exists():
                logger.info(f"Skipping {exp['name']} - results already exist.")
                continue

            print(f"RUNNING EXPERIMENT: {exp['name']}")
            logger.info(f"RUNNING EXPERIMENT: {exp['name']}")

            encoder_name = exp.get("encoder", "mit_b3")
            optimizer_name = exp.get("optimizer", "adamw_simple")
            criterion_name = exp.get("criterion", "bce_with_logits")
            size = int(exp.get("size", 512))
            is_aggressive = bool(exp.get("aggressive", False))
            any_augmentation = bool(exp.get("any_augmentation", True))
            
            if size > 512:
                batch_size = max(1, batch_size // 2)

            exp_dice_scores, exp_convergence, exp_precisions, exp_recalls = [], [], [], []

            for fold, (train_idx, val_idx) in enumerate(kf.split(full_df, full_df["stratify_label"])):
                print(f"\n--- Fold {fold+1}/{k_folds} ---")
                train_ds = Dataset(processed_dir, split="train", indices=train_idx, target_size=(size, size), more_augmentation=is_aggressive, any_augmentation=any_augmentation)
                val_ds = Dataset(processed_dir, split="val", indices=val_idx, target_size=(size, size))

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
                val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                # sanity_check_dataloader(train_loader)

                # create model and move to device
                if encoder_name == "resnet34":
                    model = get_model_resnet()
                elif encoder_name == "resnet_unet":
                    model = get_model_resnet_unet()
                elif encoder_name == "resnet_deeplabv3":
                    model = get_model_resnet_deeplabv3()
                elif encoder_name == "mit_b3_deeplabv3":
                    model = get_mit_b3_deeplabv3_model()
                else:
                    model = get_mit_b3_model()
                model.to(device)

                if optimizer_name == "adamw_simple":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
                else:
                    optimizer = torch.optim.AdamW([
                        {'params': model.encoder.parameters(), 'lr': 1e-4}, # Slow for pre-trained
                        {'params': model.decoder.parameters(), 'lr': 1e-3}, # Fast for new decoder
                        ], weight_decay=1e-1)
                
                criterion = torch.nn.BCEWithLogitsLoss()
                if criterion_name == "bce_with_logits":
                    criterion = torch.nn.BCEWithLogitsLoss()
                elif criterion_name == "hybrid_loss_0.3":
                    criterion = HybridLoss(pos_weight=weight_tensor)
                elif criterion_name == "hybrid_loss_0.2":
                    criterion = HybridLoss2(pos_weight=weight_tensor)
                elif criterion_name == "focal_dice":
                    criterion = HybridFocalDiceLoss()
                elif criterion_name == "tversky":
                    criterion = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)

                    
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=3
                )


                # train with early stopping
                logger.info(f"Starting training for {exp['name']} - Fold {fold+1}\ncriterion: {criterion_name}, optimizer: {optimizer_name}, size: {size}, aggressive_aug: {is_aggressive}")
                best_model, history = _train_with_stop(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs=epochs, patience=7)

                _save_history(history, Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{fold+1}_history.csv")
                torch.save(best_model.state_dict(), Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{fold+1}.pth")

                _, val_dice, val_precision, val_recall = validate(model, val_loader, criterion, device)

                exp_dice_scores.append(val_dice)
                exp_convergence.append(len(history["val_dice"]))
                exp_precisions.append(val_precision)
                exp_recalls.append(val_recall)
                print(f"Fold {fold+1} | Final Val Dice: {history['val_dice'][-1]:.4f} | Convergence Epoch: {len(history['val_dice'])}")
            
            log_experiment_summary(logger, exp['name'], exp_dice_scores, exp_convergence)
            
            results.append({
                "Experiment": exp['name'],
                "Dice_Mean": np.mean(exp_dice_scores),
                "Dice_Std": np.std(exp_dice_scores),
                "Dice_Scores": exp_dice_scores,
                "Precision_Mean": np.mean(exp_precisions),
                "Precision_Scores": exp_precisions,
                "Recall_Mean": np.mean(exp_recalls),
                "Recall_Scores": exp_recalls,
                "Convergence_Epochs_Mean": np.mean(exp_convergence)
            })
        except Exception as e:
            logger.error(f"Error in experiment {exp['name']}: {str(e)}")
            continue

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{results_df['Experiment'].iloc[0].lower().replace(' ', '_')}_{results_df['Experiment'].iloc[-1].lower().replace(' ', '_')}_segmentation_ablation_study_results.csv", index=False)
        print("\nAblation Study Complete. Table generated:")
        print(results_df)
        logger.info("Ablation Study Complete.")

def log_experiment_summary(logger, exp_name, fold_scores, convergence_epochs):
    """Logs a clean summary at the end of each K-Fold experiment."""
    logger.info(f"\n" + "="*40)
    logger.info(f"SUMMARY FOR {exp_name}")
    logger.info(f"Mean Dice: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    logger.info(f"Mean Convergence Epochs: {np.mean(convergence_epochs):.2f}")
    logger.info("="*40 + "\n")

def calculate_dataset_weight(df, processed_dir):
    total_bg = 0
    total_fg = 0
    print("Calculating dataset pixel ratio...")
    for _, row in df.iterrows():
        mask = cv2.imread(str(Path(processed_dir) / row.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            fg = np.sum(mask > 127)
            bg = mask.size - fg
            total_bg += bg
            total_fg += fg
    
    # Ratio of background to foreground
    ratio = total_bg / (total_fg + 1e-7)
    return ratio

import matplotlib.pyplot as plt
def sanity_check_dataloader(dataloader, num_samples=4):
    # Get one batch
    imgs, masks = next(iter(dataloader))
    
    # Move to CPU for plotting
    imgs = imgs.cpu()
    masks = masks.cpu()
    
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(imgs))):
        # 1. Denormalize Image (assuming ImageNet stats)
        img = imgs[i].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # 2. Get Mask
        mask = masks[i].squeeze().numpy() # Remove channel dim
        
        # 3. Plot
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis('off')
        
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(img)
        # Overlay mask with transparency
        plt.imshow(mask, alpha=0.5, cmap='jet') 
        plt.title(f"Overlay {i}\nMax: {mask.max()}")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()