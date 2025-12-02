import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image

def process_image(img_path, out_images, label, patient_id):
    try:
        with Image.open(img_path) as img:
            img = img.convert("L")
            img_id = f"{uuid.uuid4().hex}.png"
            img.save(out_images / img_id, optimize=True, **{"icc_profile": None, "exif": None, "pnginfo": None})
        return {
            "img_id": img_id,
            "label": label,
            "patient_id": f"{label}_{patient_id}",
            "raw_path": str(img_path)
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def extract_images(raw_dir, out_dir,workers=8):
    raw_dir = Path(raw_dir)

    out_images = Path(out_dir) / "images"
    shutil.rmtree(out_images, ignore_errors=True)
    out_images.mkdir(parents=True)

    tasks = []
    for label_dir in raw_dir.rglob("*"):
        if not label_dir.is_dir(): continue
        label = label_dir.name
        if label not in {"Benign", "Malignant", "Normal"}: continue

        for patient_dir in label_dir.iterdir():
            if not patient_dir.is_dir(): continue
            patient_id = patient_dir.name

            for img_path in patient_dir.iterdir():
                # special case for Spectra Mammos dataset
                if img_path.is_dir():
                    for img_file in img_path.iterdir():
                        if img_file.is_file():
                            tasks.append((img_file, out_images, label, patient_id))
                if img_path.is_file():
                    tasks.append((img_path, out_images, label, patient_id))

    with ThreadPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(lambda x: process_image(*x), tasks))

    df = pd.DataFrame([r for r in results if r])
    df.to_csv(Path(out_dir)/"metadata.csv", index=False)

if __name__ == "__main__":
    # LocalDataSet USG
    # raw_directory = "./data/raw/LocalDataSet/DCL_USG"
    # output_directory = "./data/processed/LocalDataSet/DCL_USG"

    # LocalDataSet Mammos
    raw_directory = "./data/raw/LocalDataSet/DCL_Mammos"
    output_directory = "./data/processed/LocalDataSet/DCL_Mammos"
    extract_images(raw_directory, output_directory)