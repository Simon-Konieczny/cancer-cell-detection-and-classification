import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import ClassificationDataset as Dataset
from cnn import BaselineCNN, ImprovedCNN, MedicalResNet
from vit import MedicalViT
from pathlib import Path
from losses import FocalLossClassification
from torch.optim.swa_utils import AveragedModel, SWALR
from train_vit import train_medical_vit
import copy

def classification(processed_dir, epochs=20, batch_size=16):
    print("Starting training...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    train_ds = Dataset(processed_dir, split="train")
    val_ds   = Dataset(processed_dir, split="val")
    train_sampler = get_balanced_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # model = MedicalViT(num_classes=3, img_size=512).to(device)

    # return train_medical_vit(model, train_loader, val_loader, device)
    model = MedicalResNet(num_classes=3).to(device)
    return _train_classification_with_stop(
        model, train_loader, val_loader, processed_dir, device,
        epochs=epochs)

def _train_classification_with_stop(model, train_loader, val_loader, processed_dir, device, epochs=50, warm_up_epochs=5, patience=12):
    # Early Stopping and Model Saving
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0
    
    # initially freeze all layers except final FC
    print("\n--- STAGE 1: Warm-Up Training (Frozen Backbone) ---\n")
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    weights = torch.tensor([0.60, 1.22, 1.98]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    criterion = FocalLossClassification(alpha=weights, gamma=3.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    swa_model = AveragedModel(model)
    swa_start = int(epochs * 0.75) # Start averaging in the last 25% of training
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

    for epoch in range(1, epochs + 1):
        # Unfreeze after warm-up
        if epoch == warm_up_epochs + 1:
            print("\n--- STAGE 2: Unfreezing Backbone & Fine-Tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW([
                {'params': model.resnet.conv1.parameters(), 'lr': 1e-6},
                {'params': model.resnet.layer1.parameters(), 'lr': 1e-6},
                {'params': model.resnet.layer2.parameters(), 'lr': 5e-6},
                {'params': model.resnet.layer3.parameters(), 'lr': 1e-5},
                {'params': model.resnet.layer4.parameters(), 'lr': 1e-5},
                {'params': model.resnet.fc.parameters(), 'lr': 1e-4}
            ], weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warm_up_epochs))
            swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
            # Reset patience for the new stage
            early_stop_counter = 0

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss, train_acc = running_loss / total, correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()

        # Update Scheduler
        if epoch <= warm_up_epochs:
            scheduler.step(val_loss) #type: ignore
        else:
            scheduler.step() #type: ignore

        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # track Val Loss because it is the best indicator of generalization
        if val_loss < best_loss:
            print(f"--> Validation Loss decreased ({best_loss:.4f} to {val_loss:.4f}). Saving best model weights.")
            best_loss = val_loss
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"--> No improvement in Val Loss. EarlyStop Counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch}. Reverting to best weights.")
            break

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    
    out_dir = Path(processed_dir) / "checkpoints"
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out_dir / f"medical_resnet_best_acc_{best_acc:.2f}.pth")

    return model

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def get_balanced_sampler(dataset):
    labels = []
    for label in dataset.df['label']:
        label_str = str(label).lower().strip()
        labels.append(dataset.label_map[label_str])
    
    class_sample_count = torch.tensor([(torch.tensor(labels) == t).sum() for t in torch.unique(torch.tensor(labels), sorted=True)])
    weight = 1. / class_sample_count.float()

    samples_weight = torch.tensor([weight[t] for t in labels])
    
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight.tolist(), num_samples=len(samples_weight), replacement=True)
    
    return sampler