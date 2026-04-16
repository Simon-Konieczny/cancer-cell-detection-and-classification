from pathlib import Path
import logging
import sys
import pandas as pd

def setup_logger(log_dir="./logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    log_file = Path(log_dir) / f"ablation_study_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter: [Timestamp] [Level] Message
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

if __name__ == "__main__":
    input = input("classification or segmentation? (c/s): ").strip().lower()
    logger = setup_logger()
    
    if input == "s":
        from seg_ablation_study import run_ablation_study
        model = run_ablation_study(
            processed_dir="./data/processed/pooled_seg_USG",
            logger=logger,
            batch_size=4,
            k_folds=5,
            epochs=100,
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