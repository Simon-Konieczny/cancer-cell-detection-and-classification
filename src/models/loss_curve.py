import os
import pandas as pd
import matplotlib.pyplot as plt

save_dir = "./loss_curves"
os.makedirs(save_dir, exist_ok=True)

def plot_training_history(csv_file, out_dir):
    df = pd.read_csv(csv_file)
    epochs = range(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Training vs Validation Loss ---
    ax1.plot(epochs, df['train_loss'], label='Training Loss', color='#2ecc71', linewidth=2)
    ax1.plot(epochs, df['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Validation F1 Score ---
    ax2.plot(epochs, df['val_f1'], label='Val F1 Score', color='#3498db', linewidth=2)
    ax2.set_title('Validation F1 Score', fontsize=14)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    save_path = os.path.join(out_dir, f"training_history_{os.path.basename(csv_file)}.png")
    plt.savefig(save_path)
    plt.close()
folder = "pooled_Mammos/"
file = "ablation_architecture:_hybrid_maxvit_fold2_history.csv"
plot_training_history("./data/processed/" + folder + file, save_dir)