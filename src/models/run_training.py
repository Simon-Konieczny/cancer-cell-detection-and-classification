from pathlib import Path
import logging
import sys
import pandas as pd

def setup_logger(log_dir="./logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a unique filename with a timestamp
    log_file = Path(log_dir) / f"ablation_study_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter: [Timestamp] [Level] Message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # File Handler (Writes to the file)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream Handler (Still prints to your terminal)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

if __name__ == "__main__":
    input = input("classification or segmentation? (c/s): ").strip().lower()
    logger = setup_logger()
    
    if input == "s":
        from train_segmentation import segmentation
        segmentation(
            processed_dir="./data/processed/pooled_seg_USG",
            epochs=100,
            batch_size=8
        )
    else:
        from class_ablation_study import run_ablation_study
        model = run_ablation_study(
            # processed_dir="./data/processed/pooled_USG",
            processed_dir="./data/processed/pooled_Mammos",
            logger=logger,
            batch_size=4,
            k_folds=5,
            epochs=75,
        )

# idea is to have three clean tables for an ablation study each on pooled_mammos, pooled_usg, and pooled_seg_usg.