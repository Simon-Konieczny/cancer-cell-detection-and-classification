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

# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
    
#     # Progress bar for dissertation monitoring
#     pbar = tqdm(loader, desc="Training")
#     for images, masks in pbar:
#         images = images.to(device)
#         masks = masks.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, masks)

#         # print(outputs.min().item(), outputs.max().item())
        
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         pbar.set_postfix(loss=loss.item())
        
#     return running_loss / len(loader)

def train_one_epoch(model, loader, optimizer, criterion, device, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    
    # IMPORTANT: Clear gradients BEFORE the loop starts
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        
        # scale the loss
        # Since we are summing gradients over 'N' steps, we divide the loss by N 
        # to keep the average gradient magnitude consistent.
        loss = criterion(outputs, masks) / accumulation_steps
        
        # Backward pass (Accumulates gradients in the buffers)
        loss.backward()

        # Only update weights every 'accumulation_steps'
        if (i + 1) % accumulation_steps == 0:
            # Gradient Clipping (essential for Transformers like MiT-B3)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad() # Finally clear the gradients for the next set

        running_loss += (loss.item() * accumulation_steps)
        pbar.set_postfix(loss=loss.item() * accumulation_steps)
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    # avoid division by zero and handle empty masks
    smooth = 1e-7
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get Logits and Probabilities
            outputs = model(images)
            # Loss calculation (on Logits)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Calculate Loss
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Vectorized Dice calculation
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            cardinality = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            
            batch_dice = (2. * intersection + smooth) / (cardinality + smooth)
            dice_scores.extend(batch_dice.cpu().tolist())

    # Calculate final averages
    metrics = {
        "loss": running_loss / len(loader),
        "dice": sum(dice_scores) / len(dice_scores)
    }
    
    return metrics

def segmentation(processed_dir, batch_size = 8, epochs=50):
    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_mit_b3_model()
    model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/best_segmentation_model_9.pth"))
    model.to(device)

    train_ds = Dataset(processed_dir, split="train")
    val_ds   = Dataset(processed_dir, split="val")

    # class_counts = np.bincount(train_ds.labels)
    # weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # sample_weights = weights[train_ds.labels]

    # sampler = WeightedRandomSampler(
    #     weights=sample_weights.tolist(), 
    #     num_samples=len(sample_weights), 
    #     replacement=True
    # )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # Optimizer & Scheduler
    # run 7 opt 
    # optimizer = torch.optim.AdamW([
    #     {'params': model.encoder.parameters(), 'lr': 1e-4}, # Slow for pre-trained
    #     {'params': model.decoder.parameters(), 'lr': 1e-3}, # Fast for new decoder
    # ], weight_decay=1e-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)

    # optimizer = torch.optim.AdamW([
    #     {'params': model.encoder.parameters(), 'lr': 2e-5},
    #     {'params': model.decoder.parameters(), 'lr': 2e-4},
    # ], weight_decay=1e-1)

    
    criterion = HybridLoss2()
    #run 7 loss
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = HybridFocalDiceLoss()
    # # criterion = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7)

    # optimizer = get_optimizer(model)

    # scheduler1 = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    # scheduler2 = CosineAnnealingLR(optimizer, T_max=45)
    # scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[5])

    # run 7 scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.2, patience=3
    # )

    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=[1e-5, 1e-5, 1e-4], # Max LR for each group defined in optimizer
    #     steps_per_epoch=len(train_loader),
    #     epochs=epochs,
    #     pct_start=0.1, # 10% of time spent warming up
    #     anneal_strategy='cos'
    # )

    # Training Loop
    best_dice = 0.0

    # Pull one batch
    # images, masks = next(iter(train_loader))

    # Visualize the first image and its mask
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(images[0][0].cpu().numpy(), cmap='gray')
    # plt.title("Image")
    # plt.subplot(1, 2, 2)
    # plt.imshow(masks[0][0].cpu().numpy(), cmap='gray')
    # plt.title("Mask")
    # plt.show()

    # print(f"Mask values: {np.unique(masks[0].cpu().numpy())}")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
    
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        
        # scheduler.step(val_metrics['dice'])
        scheduler.step(val_metrics['loss'])
        
        # Save the best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), processed_dir + "/best_segmentation_model.pth")
            print("--> Best model saved!")