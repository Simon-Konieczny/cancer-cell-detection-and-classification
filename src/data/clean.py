import pandas as pd
from PIL import Image
import imagehash
from pathlib import Path

def remove_duplicates(processed_dir, hash_threshold=5):
    images_dir = Path(processed_dir) / "images"
    metadata_path = Path(processed_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)

    seen = {}
    keep = []

    for _, row in df.iterrows():
        img_path = images_dir / row["img_id"]

        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img)
        except:
            continue

        is_duplicate = False
        for prev_hash, prev_id in seen.items():
            if abs(h - prev_hash) <= hash_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen[h] = row["img_id"]
            keep.append(row)

    df_out = pd.DataFrame(keep)
    df_out.to_csv(metadata_path, index=False)

if __name__ == "__main__":
    # processed_directory = "./data/processed/LocalDataSet/DCL_Mammos"
    processed_directory = "./data/processed/LocalDataSet/DCL_USG"
    remove_duplicates(processed_directory)