import os
import json
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from datetime import datetime
from pathlib import Path

def get_dataloaders(cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(cfg["data_dir"], transform=transform)
    val_size = int(cfg["val_split"] * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True),
        DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False),
    )

def get_model(cfg):
    model = getattr(models, cfg["model_name"])(weights="IMAGENET1K_V1" if cfg["pretrained"] else None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, cfg["num_classes"])

    if cfg["pretrained"] and not cfg["finetune"]:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            total_loss += loss.item()
            correct += (preds.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def main(config_path="train_config.yml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = get_dataloaders(cfg)
    model = get_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])

    exp_dir = Path("experiments") / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    results = []
    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        results.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch}: val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), exp_dir / "model.pt")
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
