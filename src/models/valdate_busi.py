import torch
from mit import get_mit_b3_model
from dataset import SegmentationDataset, ClassificationDataset
from torch.utils.data import DataLoader
from losses import HybridFocalDiceLoss
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from train_segmentation import validate

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()
print(f"Using device: {device}")


# model = get_mit_b3_model()
# model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/ablation_mit-b3_adam_simple_focal_fold2.pth", map_location=device))
# model.to(device)

# criterion = HybridFocalDiceLoss()

# seg_ds = SegmentationDataset("./data/processed/BUSI", split="val")
# seg_loader = DataLoader(seg_ds, batch_size=4, shuffle=False)

# results = validate(model, loader=seg_loader, criterion=criterion, device=device)

# print(results)
# df = pd.DataFrame(results)
# df.to_csv("busi_seg_validate.csv")


# model = get_mit_b3_model()
# model.load_state_dict(torch.load("./data/processed/pooled_seg_USG/ablation_resnet_adam_0.2_bce_fold2.pth", map_location=device))
# model.to(device)

from class_models import get_model
from train_classification import _validate

model = get_model(model_name="convnext_small")
model.load_state_dict(torch.load("./data/processed/pooled_USG/ablation_+_roi_cropping_fold2.pth", map_location=device))
model.to(device)

criterion = nn.CrossEntropyLoss()

class_ds = ClassificationDataset("./data/processed/BUSI", split="val")
class_loader = DataLoader(class_ds, batch_size=8, shuffle=False)

avg_loss, avg_f1, avg_auc, probs, labels = _validate(model, criterion, class_loader, device)

# 2. Get predictions by finding the index of the highest probability
preds = np.argmax(probs, axis=1)

# 3. Generate the Confusion Matrix
class_names = ["Benign", "Malignant", "Normal"]
cm = confusion_matrix(labels, preds)

# 4. Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: BUSI Validation')
plt.savefig('busi_class_validate_conf_matrix.png')
# plt.show()

# Optional: Save results to CSV (organizing the tuple into a dict first)
results_dict = {
    "loss": [avg_loss],
    "f1_macro": [avg_f1],
    "auc_macro": [avg_auc]
}
df = pd.DataFrame(results_dict)
df.to_csv("busi_class_validate_metrics.csv", index=False)


# results = _validate(model, criterion, class_loader, device)

# print(results)
# df = pd.DataFrame(results)
# df.to_csv("busi_class_validate.csv")
