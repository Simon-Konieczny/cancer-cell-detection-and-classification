import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import SegmentationDataset as Dataset
from pathlib import Path
from tqdm import tqdm
from mit import get_model_resnet, get_mit_b3_model, efficient_model
from losses import HybridLoss, HybridFocalDiceLoss, HybridLoss2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import segmentation_models_pytorch as smp
from torch.utils.data import WeightedRandomSampler
import numpy as np
from optimizer import get_optimizer
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

def _train_one_epoch(model, loader, optimizer, criterion, device, total, scaler=None):
    model.train()
    running_loss = 0.0
    
    # Progress bar for dissertation monitoring
    pbar = tqdm(loader, desc="Training")
    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler and device.type == 'cuda':
            with torch.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        total += masks.size(0)
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / total

def _train_with_stop(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs, patience):
    scaler = None
    if device.type == 'cuda':
        scaler = torch.GradScaler()
    best_model_wts = None
    best_dice = 0
    early_stop_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_precision": [],
        "val_recall": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        total = 0
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device, total, scaler)
        val_loss, val_dice, val_precision, val_recall = validate(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_model_wts = model.state_dict()
            print(f"Val Dice increased ({best_dice:.4f} --> {val_dice:.4f}). Saving best model.")
            best_dice = val_dice
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Reverting to best model weights.")
            break
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model, history

# def train_one_epoch(model, loader, optimizer, criterion, device, accumulation_steps=4):
#     model.train()
#     running_loss = 0.0
#     optimizer.zero_grad()
    
#     pbar = tqdm(loader, desc="Training")
#     for i, (images, masks) in enumerate(pbar):
#         images = images.to(device)
#         masks = masks.to(device)

#         # Forward pass
#         outputs = model(images)
        
#         # scale the loss
#         # Since we are summing gradients over 'N' steps, we divide the loss by N 
#         # to keep the average gradient magnitude consistent.
#         loss = criterion(outputs, masks) / accumulation_steps
        
#         # Backward pass (Accumulates gradients in the buffers)
#         loss.backward()

#         # Only update weights every 'accumulation_steps'
#         if (i + 1) % accumulation_steps == 0:
#             # Gradient Clipping (essential for Transformers like MiT-B3)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
#             optimizer.zero_grad() # Finally clear the gradients for the next set

#         running_loss += (loss.item() * accumulation_steps)
#         pbar.set_postfix(loss=loss.item() * accumulation_steps)
        
#     return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores, presicions, recalls = [], [], []

    smooth = 1e-7
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            dims = (1, 2, 3)
            tp = (preds * masks).sum(dim=dims)
            fp = (preds * (1 - masks)).sum(dim=dims)
            fn = ((1 - preds) * masks).sum(dim=dims)
            
            batch_dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
            batch_precision = (tp + smooth) / (tp + fp + smooth)
            batch_recall = (tp + smooth) / (tp + fn + smooth)

            dice_scores.extend(batch_dice.cpu().tolist())
            presicions.extend(batch_precision.cpu().tolist())
            recalls.extend(batch_recall.cpu().tolist())

    avg_loss = running_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_precision = np.mean(presicions)
    avg_recall = np.mean(recalls)
    
    return avg_loss, avg_dice, avg_precision, avg_recall

# def segmentation(processed_dir, batch_size = 8, epochs=50):
#     # Setup Device
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model = get_mit_b3_model()
#     model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/best_segmentation_model_9.pth"))
#     model.to(device)

#     train_ds = Dataset(processed_dir, split="train")
#     val_ds   = Dataset(processed_dir, split="val")

#     # class_counts = np.bincount(train_ds.labels)
#     # weights = 1. / torch.tensor(class_counts, dtype=torch.float)
#     # sample_weights = weights[train_ds.labels]

#     # sampler = WeightedRandomSampler(
#     #     weights=sample_weights.tolist(), 
#     #     num_samples=len(sample_weights), 
#     #     replacement=True
#     # )

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

#     # Optimizer & Scheduler
#     # run 7 opt 
#     # optimizer = torch.optim.AdamW([
#     #     {'params': model.encoder.parameters(), 'lr': 1e-4}, # Slow for pre-trained
#     #     {'params': model.decoder.parameters(), 'lr': 1e-3}, # Fast for new decoder
#     # ], weight_decay=1e-1)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

#     # optimizer = torch.optim.AdamW([
#     #     {'params': model.encoder.parameters(), 'lr': 2e-5},
#     #     {'params': model.decoder.parameters(), 'lr': 2e-4},
#     # ], weight_decay=1e-1)

    
#     criterion = HybridLoss2()
#     #run 7 loss
#     # criterion = nn.BCEWithLogitsLoss()
#     # criterion = HybridFocalDiceLoss()
#     # # criterion = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)

#     # optimizer = get_optimizer(model)

#     # scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
#     # scheduler2 = CosineAnnealingLR(optimizer, T_max=45)
#     # scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])

#     # run 7 scheduler
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=3
#     )

#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     #     optimizer, mode='max', factor=0.2, patience=3
#     # )

#     # scheduler = OneCycleLR(
#     #     optimizer,
#     #     max_lr=[1e-5, 1e-5, 1e-4], # Max LR for each group defined in optimizer
#     #     steps_per_epoch=len(train_loader),
#     #     epochs=epochs,
#     #     pct_start=0.1, # 10% of time spent warming up
#     #     anneal_strategy='cos'
#     # )

#     # Training Loop
#     best_dice = 0.0

#     # Pull one batch
#     # images, masks = next(iter(train_loader))

#     # Visualize the first image and its mask
#     # plt.figure(figsize=(10, 5))
#     # plt.subplot(1, 2, 1)
#     # plt.imshow(images[0][0].cpu().numpy(), cmap='gray')
#     # plt.title("Image")
#     # plt.subplot(1, 2, 2)
#     # plt.imshow(masks[0][0].cpu().numpy(), cmap='gray')
#     # plt.title("Mask")
#     # plt.show()

#     # print(f"Mask values: {np.unique(masks[0].cpu().numpy())}")

#     for epoch in range(epochs):
        # print(f"\nEpoch {epoch+1}/{epochs}")
        
        # train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # val_metrics = validate(model, val_loader, criterion, device)
    
        # print(f"Epoch {epoch+1}")
        # print(f"Train Loss: {train_loss:.4f}")
        # print(f"Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        
        # # scheduler.step(val_metrics['dice'])
        # scheduler.step(val_metrics['loss'])
        
        # # Save the best model
        # if val_metrics['dice'] > best_dice:
        #     best_dice = val_metrics['dice']
        #     torch.save(model.state_dict(), processed_dir + "/best_segmentation_model.pth")
        #     print("--> Best model saved!")