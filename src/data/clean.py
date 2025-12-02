from imagehash import phash
from PIL import Image
from pathlib import Path
import pandas as pd

def remove_duplicates(processed_dir, threshold=5):
    images_dir = Path(processed_dir) / "images"
    df = pd.read_csv(Path(processed_dir) / "metadata.csv")

    buckets = {}
    keep = []

    for _, row in df.iterrows():
        img_path = images_dir / row["img_id"]
        try:
            with Image.open(img_path) as img:
                h = phash(img)
        except:
            continue

        h_int = int(str(h), 16)
        bucket = h_int >> 8

        is_dup = False
        if bucket in buckets:
            for prev in buckets[bucket]:
                if abs(h - prev["hash"]) <= threshold:
                    is_dup = True
                    break

        if not is_dup:
            buckets.setdefault(bucket, []).append({"hash": h, "id": row["img_id"]})
            keep.append(row)

    pd.DataFrame(keep).to_csv(Path(processed_dir)/"metadata.csv", index=False)