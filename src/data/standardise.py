import cv2
import pandas as pd
from pathlib import Path

def process_image(img_path, resize=(224, 224), clahe=False, denoise=False):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    if denoise:
        img = cv2.medianBlur(img, 3)

    if clahe:
        clahe_fn = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe_fn.apply(img)

    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    return img

def apply_standardisation(processed_dir, resize=(224, 224), clahe=False, denoise=False):
    images_dir = Path(processed_dir) / "images"
    metadata_path = Path(processed_dir) / "metadata.csv"
    df = pd.read_csv(metadata_path)

    if clahe:
        clahe_fn = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i, row in df.iterrows():
        img_path = images_dir / row["img_id"]

        img = process_image(img_path, resize=resize, clahe=clahe, denoise=denoise)
        
        if img is None:
            print(f"Warning: could not process image {img_path}, skipping.")
            continue

        cv2.imwrite(str(img_path), img)
        df.loc[i, "height"] = img.shape[0]
        df.loc[i, "width"] = img.shape[1]

    df.to_csv(metadata_path, index=False)

if __name__ == "__main__":
    processed_directory = "./data/processed/LocalDataSet/DCL_Mammos"
    # processed_directory = "./data/processed/LocalDataSet/DCL_USG"
    apply_standardisation(processed_directory, resize=(224, 224), clahe=True, denoise=True)