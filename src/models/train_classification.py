import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import ClassificationDataset as Dataset
from cnn import BaselineCNN, ImprovedCNN, MedicalResNet
from pathlib import Path
import copy

def classification(processed_dir, epochs=20, batch_size=16, lr=1e-3):
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = MedicalResNet(num_classes=3).to(device)

    weights = torch.tensor([0.60, 1.22, 1.98]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3
    )

    return _train_classification_with_stop(
        model, train_loader, val_loader, criterion, processed_dir, device,
        epochs=epochs, warm_up_epochs=10, patience=7)

def _train_classification_with_stop(model, train_loader, val_loader, criterion, processed_dir, device, epochs=50, warm_up_epochs=5, patience=7):
    # Early Stopping and Model Saving
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    early_stop_counter = 0
    
    # initially freeze all layers except final FC
    print("\n--- STAGE 1: Warm-Up Training (Frozen Backbone) ---")
    for param in model.resnet.parameters():
        param.requires_grad = False
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    for epoch in range(1, epochs + 1):
        # Unfreeze after warm-up
        if epoch == warm_up_epochs + 1:
            print("\n--- STAGE 2: Unfreezing Backbone & Fine-Tuning ---")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warm_up_epochs))
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