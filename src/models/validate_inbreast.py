import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.swa_utils import AveragedModel

from class_models import get_model
from train_classification import _validate
from losses import FocalLossClassification
from dataset import ClassificationDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
elif device.type == 'mps':
    torch.mps.empty_cache()
print(f"Using device: {device}")

model = get_model(model_name="convnext_small")
model = AveragedModel(model)
model.load_state_dict(torch.load("./data/processed/pooled_Mammos/ablation_final_(roi+focal+swa)_fold2.pth", map_location=device))
model.to(device)

ds = ClassificationDataset('./data/processed/INBREAST', split='val')
loader = DataLoader(ds, batch_size=8, shuffle=False)

weights = ds.weights
weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = FocalLossClassification(alpha=weights)

avg_loss, avg_f1, avg_auc, probs, labels = _validate(model, criterion, loader, device)

MALIGNANT_IDX = 1
MALIGNANT_THRESHOLD = 0.25

preds = []
for prob in probs:
    if prob[MALIGNANT_IDX] >= MALIGNANT_THRESHOLD:
        preds.append(MALIGNANT_IDX)
    else:
        # among the remaining classes, pick the highest
        preds.append(np.argmax(prob))
preds = np.array(preds)

# preds = np.argmax(probs, axis=1)

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
plt.title('Confusion Matrix: INBREAST Validation')
plt.savefig('inbreast_class_validate_conf_matrix.png')

results_dict = {
    "loss": [avg_loss],
    "f1_macro": [avg_f1],
    "auc_macro": [avg_auc]
}
df = pd.DataFrame(results_dict)
df.to_csv("inbreast_class_validate_metrics.csv", index=False)