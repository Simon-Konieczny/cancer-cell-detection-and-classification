import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_history(csv_file, out_dir):
    """Generates and saves training plots from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        epochs = range(1, len(df) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # --- Plot 1: Training vs Validation Loss ---
        ax1.plot(epochs, df['train_loss'], label='Training Loss', color='#2ecc71', linewidth=2)
        ax1.plot(epochs, df['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
        ax1.set_title(f'Model Loss: {csv_file.name}', fontsize=12)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- Plot 2: Validation Dice, Precision, and Recall ---
        ax2.plot(epochs, df['val_dice'], label='Val Dice Score', color='#3498db', linewidth=2)
        ax2.plot(epochs, df['val_precision'], label='Val Precision', color="#34db7c", linewidth=2)
        ax2.plot(epochs, df['val_recall'], label='Val Recall', color="#7734db", linewidth=2)
        ax2.set_title('Validation Scores', fontsize=12)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

        # Save using the stem (filename without extension)
        save_path = out_dir / f"plot_{csv_file.stem}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Successfully plotted: {csv_file.name}")
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

def main(input_root, output_root):
    # Convert strings to Path objects
    src_path = Path(input_root)
    dest_path = Path(output_root)

    # Create output directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Use rglob for recursive search (change to .glob if you only want the top level)
    csv_files = list(src_path.rglob('*_history.csv'))

    if not csv_files:
        print("No matching CSV files found.")
        return

    print(f"Found {len(csv_files)} files. Starting plotting...")

    for csv_path in csv_files:
        plot_training_history(csv_path, dest_path)

if __name__ == "__main__":
    # Set your directories here
    INPUT_DIRECTORY = './data/processed/pooled_seg_USG' 
    OUTPUT_DIRECTORY = './segmentation_visuals/plots'
    
    main(INPUT_DIRECTORY, OUTPUT_DIRECTORY)