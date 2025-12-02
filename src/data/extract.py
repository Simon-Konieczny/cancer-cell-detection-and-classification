import shutil
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image

def extract_images(raw_dir, out_dir):
    raw_dir = Path(raw_dir)

    out_images = Path(out_dir) / "images"
    if out_images.exists():
        shutil.rmtree(out_images)

    out_images.mkdir(parents=True, exist_ok=False)

    records = []

    for label_dir in raw_dir.rglob("*"):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        if label not in {"Benign", "Malignant", "Normal"}:
            continue

        for patient_dir in label_dir.iterdir():
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            for img_path in patient_dir.iterdir():
                if not img_path.is_file():
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                except Exception:
                    print(f"Skipping unreadable file: {img_path}")
                    continue

                new_id = f"{uuid.uuid4().hex}.png"
                new_path = out_images / new_id
                img.save(new_path)

                records.append({
                    "img_id": new_id,
                    "label": label,
                    "patient_id": label + "_" + patient_id,
                    "raw_path": str(img_path)
                })

    df = pd.DataFrame(records)
    df.to_csv(Path(out_dir) / "metadata.csv", index=False)
    print("Extraction complete:", len(df), "images")

if __name__ == "__main__":
    # LocalDataSet USG
    # raw_directory = "./data/raw/LocalDataSet/DCL_USG"
    # output_directory = "./data/processed/LocalDataSet/DCL_USG"

    # LocalDataSet Mammos
    raw_directory = "./data/raw/LocalDataSet/DCL_Mammos"
    output_directory = "./data/processed/LocalDataSet/DCL_Mammos"
    extract_images(raw_directory, output_directory)