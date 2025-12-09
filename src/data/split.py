import pandas as pd
import numpy as np
from pathlib import Path

def split_dataset(processed_dir, val_ratio=0.10, test_ratio=0.10, by='patient_id'):
    metadata_path = Path(processed_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)

    unique_ids = df[by].unique()
    np.random.shuffle(unique_ids)

    n = len(unique_ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = set(unique_ids[:n_test])
    val_ids  = set(unique_ids[n_test:n_test+n_val])

    def assign_split(pid):
        if pid in test_ids: return "test"
        if pid in val_ids: return "val"
        return "train"

    df["split"] = df[by].apply(assign_split)
    df.to_csv(metadata_path, index=False)


def split_cla_dataset(processed_dir, val_ratio=0.10, test_ratio=0.10):
    split_dataset(processed_dir, val_ratio, test_ratio, by='patient_id')

def split_seg_dataset(processed_dir, val_ratio=0.10, test_ratio=0.10):
    split_dataset(processed_dir, val_ratio, test_ratio, by='base_name')