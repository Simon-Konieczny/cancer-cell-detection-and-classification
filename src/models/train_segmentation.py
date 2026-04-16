import torch
from tqdm import tqdm
import numpy as np

def _train_one_epoch(model, loader, optimizer, criterion, device, total, scaler=None):
    model.train()
    running_loss = 0.0
    
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