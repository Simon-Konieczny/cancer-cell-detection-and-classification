import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from standardise import apply_standardization

def create_row(**kwargs):
    return {k: v for k, v in kwargs.items() if v not in ("", None)}

def extract_classification(raw_dir, out_dir,workers=8):
    def process_image(img_path, out_path, label, patient_id, resize=(224, 224), clahe=False, denoise=False):
        try:
            img_id = f"{uuid.uuid4().hex}.png"
            img_hash = apply_standardization(img_path, out_path / img_id, resize=resize, clahe=clahe, denoise=denoise)
            if img_hash is None:
                print(f"Warning: could not process image {img_path}, skipping.")
                return None
            
            return create_row(img_id=img_id,
                              img_hash=img_hash,
                              label=label,
                              patient_id=f"{label}_{patient_id}",
                              raw_path=str(img_path),
                              height=resize[0],
                              width=resize[1])
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
        base, paths, out_images, out_masks, height, width = entry
        img_path = paths.get("img")
        mask_path = paths.get("mask")

        if img_path is None or mask_path is None:
            return None

        pair_id = uuid.uuid4().hex

        out_img = out_images / f"{pair_id}.png"
        out_mask = out_masks / f"{pair_id}.png"

        try:
            img_hash = apply_standardization(img_path, out_img, resize=(height, width), clahe=True, denoise=True)
            apply_standardization(mask_path, out_mask, resize=(height, width), clahe=False, denoise=False)
        except Exception as e:
            print(f"Error processing pair {base}: {e}")
            return None

        # for segmentation and classification combined datasets
        label = None
        if img_path:
            candidate = img_path.parent.name
            if candidate in {"benign", "malignant", "normal"}:
                label = candidate

        result = create_row(img_id=f"{pair_id}.png",
                            img_hash=img_hash,
                            base_name=base,
                            img_path=f"images/{pair_id}.png",
                            mask_path=f"masks/{pair_id}.png",
                            raw_img=str(img_path),
                            raw_mask=str(mask_path),
                            height=height,
                            width=width,
                            label=label)
    
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

    tasks = [(base, paths, out_images, out_masks, 224, 224) for base, paths in pairs.items()]

    with ThreadPoolExecutor(max_workers=workers) as exe:
        results = list(exe.map(copy_pair, tasks))

    df = pd.DataFrame([r for r in results if r])
    df.to_csv(out_dir / "metadata.csv", index=False)