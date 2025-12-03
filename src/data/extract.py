import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image

def extract_classification(raw_dir, out_dir,workers=8):
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
    
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    out_images = out_dir / "images"
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
    df.to_csv(out_dir / "metadata.csv", index=False)

def extract_segmentation(raw_dir, out_dir, workers=8):
    def copy_pair(entry):
        base, paths, out_images, out_masks = entry
        img_path = paths.get("img")
        mask_path = paths.get("mask")

        if img_path is None or mask_path is None:
            return None

        pair_id = uuid.uuid4().hex

        out_img = out_images / f"{pair_id}.png"
        out_mask = out_masks / f"{pair_id}.png"

        shutil.copy2(img_path, out_img)
        shutil.copy2(mask_path, out_mask)

        result = {
            "img_id": f"{pair_id}.png",
            "base_name": base,
            "img_path": f"images/{pair_id}.png",
            "mask_path": f"masks/{pair_id}.png",
            "raw_img": str(img_path),
            "raw_mask": str(mask_path),
        }

        # for segmentation and clssification combined datasets
        label = img_path.parent.name if img_path else None
        if label in {"benign", "malignant", "normal"}:
            result["label"] = label
    
        return result

    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    shutil.rmtree(out_images, ignore_errors=True)
    shutil.rmtree(out_masks, ignore_errors=True)
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    pairs = {}
    for p in raw_dir.rglob("*.png"):
        name = p.stem
        if name.endswith("_mask"):
            base = name[:-5]
            pairs.setdefault(base, {})["mask"] = p
        elif name.endswith("_tumor"):
            base = name[:-6]
            pairs.setdefault(base, {})["mask"] = p
        else:
            base = name
            pairs.setdefault(base, {})["img"] = p

    tasks = [(base, paths, out_images, out_masks) for base, paths in pairs.items()]

    with ThreadPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(copy_pair, tasks))

    df = pd.DataFrame([r for r in results if r])
    df.to_csv(out_dir / "metadata.csv", index=False)