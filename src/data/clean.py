from imagehash import phash
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np

def dedupe(processed_dir, threshold=5, bucket_shift=4):
    df = pd.read_csv(Path(processed_dir) / "metadata.csv")

    hashes = df["img_hash"].to_numpy(dtype=np.uint64)

    buckets = hashes >> bucket_shift
    df["bucket"] = buckets

    keep_mask = np.zeros(len(df), dtype=bool)

    grouped = df.groupby("bucket")

    kept_hashes = {}

    for bucket, group in grouped:
        group_hashes = group["img_hash"].to_numpy(dtype=np.uint64)
        group_indices = group.index.to_numpy()

        if bucket not in kept_hashes:
            kept_hashes[bucket] = []

        for h, idx in zip(group_hashes, group_indices):
            prev = kept_hashes[bucket]

            if len(prev) > 0:
                if np.any(np.abs(prev - h) <= threshold):
                    continue

            kept_hashes[bucket] = np.append(prev, h)
            keep_mask[df.index == idx] = True

    df[keep_mask].drop(columns=["bucket"]).to_csv(
        Path(processed_dir) / "metadata.csv", index=False
    )