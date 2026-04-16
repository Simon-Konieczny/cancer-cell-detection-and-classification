import torch
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import pandas as pd

def _train_with_stop(model, optimizer, criterion, scheduler, train_loader, val_loader, device, epochs, warm_up_epochs, patience):
    best_f1, val_f1 = 0.0, 0.0
    early_stop_counter = 0
    best_model_wts = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            for name, param in model.named_parameters():
                param.requires_grad = ("classifier" in name or "head" in name)
        elif epoch == warm_up_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            print("--- Warm-up finished: Unfreezing all layers ---")
        correct, total = 0, 0
        scaler=None
        if device.type == 'cuda':
            scaler = torch.GradScaler()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, correct, total, scaler=scaler)
        val_loss, val_f1, avg_auc, all_probs, all_labels = _validate(model, criterion, val_loader, device, best_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_f1)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # track Val F1 because it is the best indicator of generalization
        if val_f1 > best_f1:
            best_model_wts = model.state_dict()
            print(f"--> Val F1 increased ({best_f1:.4f} to {val_f1:.4f}). Saving model.")
            best_f1 = val_f1
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"--> No improvement in Val F1. EarlyStop Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. Reverting to best weights.")
            break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return model, history

def train_one_epoch(model, loader, optimizer, criterion, device, correct, total, scaler=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler and device.type == 'cuda':
            with torch.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def _train_with_swa(model, optimizer, criterion, standard_scheduler, train_loader, val_loader, device, epochs, logger):
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
    swa_start = int(epochs * 0.75) # Start SWA at 75% of training
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "lr": [],
        "is_swa": []
    }

    best_f1, val_f1 = 0.0, 0.0

    for epoch in range(1, epochs + 1):
        correct, total = 0, 0
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, correct, total)
        val_loss, val_f1, _, _, _ = _validate(model, criterion, val_loader, device, best_f1)
        logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["is_swa"].append(epoch > swa_start)
        
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            standard_scheduler.step(val_f1)

    update_bn(train_loader, swa_model, device=device)
    _, final_f1, final_auc, _, _ = _validate(swa_model, criterion, val_loader, device)
    print(f"Final SWA Model Results -> F1: {final_f1:.4f} | AUC: {final_auc:.4f}")
    return swa_model, history

def _validate(model, criterion, loader, device, best_f1=None):
    model.eval()
    running_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast('cuda'):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * imgs.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
        else:
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    avg_loss = running_loss / len(loader.dataset)
    avg_f1 = f1_score(all_labels_np, all_preds_np, average='macro')
    # plot_confusion_matrix(all_labels, all_preds)
    
    all_labels_bin = label_binarize(all_labels_np, classes=[0, 1, 2])
    
    unique_classes = np.unique(all_labels_np)
    if len(unique_classes) > 1:
        try:
            avg_auc = roc_auc_score(all_labels_bin, all_probs_np, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"AUC Error: {e}")
            avg_auc = 0.0
    else:
        print("Warning: Only one class present in validation split. AUC set to 0.5")
        avg_auc = 0.5

    if best_f1 and avg_f1 > best_f1:
        print("\n[Detailed Report]")
        print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant", "Normal"]))
    return avg_loss, avg_f1, avg_auc, np.array(all_probs), np.array(all_labels)

def _find_best_thresholds(all_probs, all_labels):
    """
    all_probs: [N, 3] array of softmax probabilities
    all_labels: [N] array of true indices
    """
    best_f1 = 0
    best_weights = [1.0, 1.0, 1.0]
    
    # Search range for multipliers
    search_space = np.linspace(0.8, 1.5, 8) 

    for w_benign in search_space:
        for w_malignant in search_space:
            current_weights = np.array([w_benign, w_malignant, 1.0])
            weighted_probs = all_probs * current_weights
            preds = np.argmax(weighted_probs, axis=1)
            
            f1 = f1_score(all_labels, preds, average='macro')
            
            if f1 > best_f1:
                best_f1 = f1
                best_weights = current_weights
                
    print(f"--- Optimization Complete ---")
    print(f"Best Macro F1: {best_f1:.4f}")
    print(f"Best Bias Weights (B, M, N): {best_weights}")
    return best_weights

def _tta_validate(model, loader, device, best_biases=None):
    model.eval()
    all_probs, all_labels = [], []

    print("Running Inference with TTA...")
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            
            logits1 = model(imgs)
            
            logits2 = model(torch.flip(imgs, dims=[3]))
            
            logits3 = model(torch.flip(imgs, dims=[2]))

            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            probs3 = F.softmax(logits3, dim=1)
            
            avg_probs = (probs1 + probs2 + probs3) / 3.0
            
            all_probs.append(avg_probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    if best_biases is not None:
        all_probs = all_probs * best_biases
        
    preds = np.argmax(all_probs, axis=1)
    return all_probs, all_labels, preds

def _save_history(history, path):
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)