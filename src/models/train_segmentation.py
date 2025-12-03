import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import SegmentationDataset as Dataset
from pathlib import Path
from baseline_unet import UNet

def train_segmentation(processed_dir, epochs=20, batch_size=4, lr=1e-3, num_classes=3):
    print("Starting segmentation training...")

    # Device
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

    model = UNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0
        total = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)             # [N,1,H,W]
            masks = masks.squeeze(1).long().to(device)  # [N,H,W]

            optimizer.zero_grad()
            outputs = model(imgs)              # [N,C,H,W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total += masks.size(0)

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / total

        val_loss, val_iou = evaluate_seg(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{epochs} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| Val IoU: {val_iou:.4f}")

    out_dir = Path(processed_dir) / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "unet_final.pth")

    return model

def evaluate_seg(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.squeeze(1).long().to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(1)  # [N,H,W]

            # Intersection over Union
            iou = compute_iou(preds, masks)
            total_iou += iou
            num_batches += 1

    return running_loss / len(loader.dataset), total_iou / num_batches

def compute_iou(pred, target, num_classes=3):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    return sum(ious) / len(ious) if ious else 0