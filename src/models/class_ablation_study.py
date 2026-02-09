import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path

from dataset import ClassificationDataset as Dataset
from class_models import get_model
from losses import FocalLossClassification
from train_classification import _validate, _train_with_stop, _train_with_swa, _find_best_thresholds, _tta_validate, _save_history


def run_ablation_study(processed_dir, logger, batch_size, k_folds, epochs):
    root = Path(processed_dir)
    results = []
    full_df = pd.read_csv(root / "metadata.csv")
    label_map = {"benign": 0, "malignant": 1, "normal": 2}
    full_df["label_idx"] = full_df["label"].str.lower().map(label_map)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    print(f"Using device: {device}")

    # define experiments
    experiments = [
    # Block 1: The "Evolution" (Baseline to Final)
    # {"name": "Baseline (ConvNeXt + CE)", "roi": False, "loss": "CE", "swa": False, "model": "convnext_small"},
    # {"name": "+ ROI Cropping", "roi": True, "loss": "CE", "swa": False, "model": "convnext_small"},
    {"name": "+ Focal Loss", "roi": True, "loss": "Focal", "swa": False, "model": "convnext_small"},
    
    # Block 2: Architectural Comparison
    # {"name": "Architecture: Hybrid_MaxViT", "roi": True, "loss": "Focal", "swa": False, "model": "maxvit_tiny_tf_512"},
    # {"name": "ViT_Swin_Tiny", "model": "swin_tiny_patch4_window7_224", "roi": True, "loss": "Focal", "swa": False},
    # {"name": "Architecture: EfficientNetV2-S", "roi": True, "loss": "Focal", "swa": False, "model": "efficientnetv2_rw_s"},
    
    # Block 3: Final Proposed Model
    {"name": "Final (ROI+Focal+SWA)", "roi": True, "loss": "Focal", "swa": True, "model": "convnext_small"},
]

    for exp in experiments:
        try:
            final_model_path = Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{k_folds}.pth"
        
            if final_model_path.exists():
                logger.info(f"Skipping {exp['name']} - results already exist.")
                continue

            print(f"RUNNING EXPERIMENT: {exp['name']}")
            logger.info(f"RUNNING EXPERIMENT: {exp['name']}")

            model_name = exp.get("model", "convnext_small")
            img_size = 672 if model_name.startswith("swin_tiny") else 640

            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            exp_f1s, exp_aucs = [], []

            for fold, (train_idx, val_idx) in enumerate(kf.split(full_df, full_df["label_idx"])):
                # create fold datasets
                train_ds = Dataset(processed_dir, split="train", indices=train_idx, use_roi=exp['roi'], img_size=img_size)
                val_ds = Dataset(processed_dir, split="val", indices=val_idx, use_roi=exp['roi'], img_size=img_size)

                weights = train_ds.weights
                print(f"Class weights: {weights}")
                weights = torch.tensor(weights, dtype=torch.float32).to(device)

                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=batch_size)

                model = get_model(model_name=model_name, input_size=img_size).to(device)
                classifier_params = []
                backbone_params = []

                for name, param in model.named_parameters():
                    if "head" in name or "classifier" in name:
                        classifier_params.append(param)
                    else:
                        backbone_params.append(param)

                optimizer = optim.AdamW([
                    {'params': classifier_params, 'lr': 1e-3},
                    {'params': backbone_params, 'lr': 1e-5}
                ], weight_decay=0.05)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=0.001)

                criterion = FocalLossClassification(alpha=weights) if exp['loss'] == "Focal" else nn.CrossEntropyLoss()

                if exp['swa']:
                    trained_model, history = _train_with_swa(model, optimizer, criterion, scheduler, train_loader, val_loader, device, 45, logger)
                else:
                    trained_model, history = _train_with_stop(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs, warm_up_epochs=5, patience=12)

                # save training history
                _save_history(history, Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{fold+1}_history.csv")

                # evaluate this fold
                _, fold_f1,auc, all_probs, all_labels = _validate(trained_model, criterion, val_loader, device)
                best_bias_weights = _find_best_thresholds(all_probs, all_labels)
                _, final_labels, final_preds = _tta_validate(trained_model, val_loader, device, best_bias_weights)
                torch.save(trained_model.state_dict(), Path(processed_dir) / f"ablation_{exp['name'].lower().replace(' ', '_')}_fold{fold+1}.pth")
                exp_f1s.append(fold_f1)
                exp_aucs.append(auc)
                print(f"Fold {fold+1} F1: {fold_f1:.4f}")
                print("\n[FINAL OPTIMIZED REPORT]")
                print(classification_report(final_labels, final_preds, target_names=["Benign", "Malignant", "Normal"]))
                logger.info(f"Fold {fold+1} complete.")

            log_experiment_summary(logger, exp['name'], exp_f1s, exp_aucs)

            results.append({
                "Experiment": exp['name'],
                "F1_Mean": np.mean(exp_f1s),
                "F1_Std": np.std(exp_f1s),
                "AUC_Mean": np.mean(exp_aucs)
            })

        except Exception as e:
                logger.error(f"Error during experiment {exp['name']}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        results_df.to_csv("ablation_study_results.csv", index=False)
        print("\nAblation Study Complete. Table generated:")
        print(results_df)
        logger.info("Ablation Study Complete.")

def log_experiment_summary(logger, exp_name, fold_scores, fold_aucs):
    """Logs a clean summary at the end of each K-Fold experiment."""
    logger.info(f"\n" + "="*40)
    logger.info(f"SUMMARY FOR {exp_name}")
    logger.info(f"Mean F1: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    logger.info(f"Mean AUC: {np.mean(fold_aucs):.4f}")
    logger.info("="*40 + "\n")