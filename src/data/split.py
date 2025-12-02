import pandas as pd
import numpy as np
from pathlib import Path

def split_cla_dataset(processed_dir, val_ratio=0.10, test_ratio=0.10):
    metadata_path = Path(processed_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)

    patients = df["patient_id"].unique()
    np.random.shuffle(patients)

    n = len(patients)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_patients = set(patients[:n_test])
    val_patients  = set(patients[n_test:n_test+n_val])

    def assign_split(pid):
        if pid in test_patients: return "test"
        if pid in val_patients: return "val"
        return "train"

    df["split"] = df["patient_id"].apply(assign_split)
    df.to_csv(metadata_path, index=False)

def split_seg_dataset(processed_dir, val_ratio=0.10, test_ratio=0.10):
    metadata_path = Path(processed_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)

    cases = df["base_name"].unique()
    np.random.shuffle(cases)

    n = len(cases)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_cases = set(cases[:n_test])
    val_cases  = set(cases[n_test:n_test+n_val])

    def assign_split(pid):
        if pid in test_cases: return "test"
        if pid in val_cases: return "val"
        return "train"

    df["split"] = df["base_name"].apply(assign_split)
    df.to_csv(metadata_path, index=False)