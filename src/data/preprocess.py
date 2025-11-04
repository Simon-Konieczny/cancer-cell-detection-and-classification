import os
import shutil
from pathlib import Path
from PIL import Image
import yaml

def preprocess_image(img_path, output_path, size=(224, 224)):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        img.save(output_path)

def main(config_path="config.yml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    src_root = Path(cfg["data_root"])
    dst_root = Path(cfg["output_dir"])
    dst_root.mkdir(parents=True, exist_ok=True)

    for folder in cfg["subfolders"]:
        src_dir = src_root / folder
        for img_file in src_dir.glob("*.jpg"):
            out_dir = dst_root / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / img_file.name
            preprocess_image(img_file, out_path)

if __name__ == "__main__":
    main()
