# Detection and Classification of Cancer Cells in Breast Ultrasound Images

This repository contains the full source code and experimental framework for a dissertation focused on the automated detection and classification of cancer cells using deep learning. The project evaluates multiple state-of-the-art architectures and optimization techniques across both ultrasound (USG) and mammography datasets.

---

## 📂 Project Structure

The repository is organized to facilitate reproducibility and clarity for academic review:

* **`src/data/`**: Pipeline for data ingestion, including standardization (CLAHE, denoising), deduplication using perceptual hashing (pHash), and stratified splitting by patient ID.
* **`src/models/`**: Core implementation of classification and segmentation models, training loops (incorporating SWA and Early Stopping), and loss functions (e.g., Focal Loss, Hybrid Focal-Dice).
* **`src/exploration/`**: Analytical tools for generating Hedge’s $g$ statistical heatmaps, loss curves, and box plots to evaluate model performance across 5-fold cross-validation.
* **`environment.yml`**: Full Conda environment specification (Python 3.11).

---

## 🛠️ Installation & Setup

### Environment Creation
To ensure a consistent environment for grading, please use the provided configuration:

```bash
conda env create -f environment.yml
conda activate cancer-detection
```

### Hardware Support
The codebase automatically detect and utlizes MPS for Apple Sillicon, CUDA for NVIDIA GPUs, or defaults to CPU.

# 🚀 Execution Pipelines
1. Data Preprocessing
    The pipeline processes raw images into a standardised format ready for training:
    - Command: `python ./src/data/run.py`.
    - Operations: Includes black-background artifact removal, ROI (Region of Interest) cropping, and image resizing to 672x672 or 640x640 depending on the model.
    - Output: Processed data is stored in `./data/processed/`.

2. Model Training & Ablation Studies
    The repository supports automated ablation studies to compare various techniques:
    - Command: python `./src/models/run_training.py`.
    - Methodology: Supports K-fold cross-validation and allows selection between classification (c) and segmentation (s) tasks.

# 📊 Methodology Highlights
- Architectures Evaluated:
    - Classification: EfficientNetV2-S, ViT Swin Tiny, Hybrid MaxViT, and ConvNeXt.
    - Segmentation: U-Net/U-Net++ with ResNet34 and MiT-B3 (SegFormer) backbones.
- Key Techniques:
    - Optimization: Stochastic Weight Averaging (SWA) and AdamW with OneCycleLR scheduling.
    - Loss Functions: Focal Loss to address class imbalance and Hybrid Dice-BCE loss for precise segmentation.
- Explainability: Integration of Grad-CAM++ to visualize model attention on malignant vs. benign features, visualization of segmentation model performance.

# 📈 Summary of Results
Detailed results are available in the root directory via .csv files. Benchmark results include:
- BUSI
    - Classification: F1 Macro 0.96 AUC 0.9894
    - Segmentation: DSC 0.90 Precision 0.92
- INbreast
    - Classification: AUC of 0.65

--------------------------------------------------------------------------------
This repository was developed as part of a dissertation project. All code and results are original unless cited otherwise.