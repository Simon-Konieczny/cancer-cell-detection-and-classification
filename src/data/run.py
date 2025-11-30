from extract import extract_images
from clean import remove_duplicates
from standardise import apply_standardisation
from split import split_dataset

def run_pipeline(raw="data/raw", out="data/processed"):
    extract_images(raw, out)
    remove_duplicates(out)
    apply_standardisation(out, resize=(224, 224), clahe=True, denoise=True)
    split_dataset(out)

if __name__ == "__main__":
    localDataSet = [("./data/raw/LocalDataSet/DCL_Mammos", "./data/processed/LocalDataSet/DCL_Mammos"),
                    ("./data/raw/LocalDataSet/DCL_USG", "./data/processed/LocalDataSet/DCL_USG")]
    for raw_dir, out_dir in localDataSet:
        run_pipeline(raw_dir, out_dir)