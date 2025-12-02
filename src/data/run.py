from extract import extract_images
from clean import remove_duplicates
from standardise import apply_standardisation
from concurrent.futures import ProcessPoolExecutor
from split import split_dataset

def run_pipeline(raw, out):
    print(f"Running pipeline:\n Raw data: {raw}\n Processed data: {out}\n")
    extract_images(raw, out)
    remove_duplicates(out)
    apply_standardisation(out, resize=(224, 224), clahe=True, denoise=True)
    split_dataset(out)
    print(f"Pipeline complete for raw: {raw}.")

def run_all(dataset, workers=5):
    tasks = []
    for raw_dir, out_dir in dataset:
        tasks.append((raw_dir, out_dir))

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(run_pipeline, inp, out) for inp, out in tasks]
        for f in futures:
            f.result()

if __name__ == "__main__":
    localDataSet = [("./data/raw/LocalDataSet/DCL_Mammos", "./data/processed/LocalDataSet/DCL_Mammos"),
                    ("./data/raw/LocalDataSet/DCL_USG", "./data/processed/LocalDataSet/DCL_USG"),
                    ("./data/raw/LocalDataSet/Spectra_Mammos", "./data/processed/LocalDataSet/Spectra_Mammos")]
    run_all(localDataSet)