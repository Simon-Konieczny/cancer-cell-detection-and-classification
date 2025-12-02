from extract import extract_classification, extract_segmentation
from clean import remove_duplicates
from standardise import apply_standardisation
from concurrent.futures import ProcessPoolExecutor
from split import split_cla_dataset, split_seg_dataset

COMMON_STEPS = [
    remove_duplicates,
    lambda out: apply_standardisation(out, resize=(224, 224), clahe=True, denoise=True),
]

def _run_single_pipeline(raw: str, out: str, extract_func, split_func):
    """Executes a complete pipeline for a single raw/out pair."""
    print(f"Starting pipeline:\n Raw data: {raw}\n Processed data: {out}")
    
    extract_func(raw, out)
    
    for step in COMMON_STEPS:
        step(out)

    split_func(out)
    
    print(f"Pipeline complete for raw: {raw}")

def run_classification_pipeline(raw, out):
    _run_single_pipeline(raw, out, extract_classification, split_cla_dataset)

def run_segmentation_pipeline(raw, out):
    _run_single_pipeline(raw, out, extract_segmentation, split_seg_dataset)

def run_all(data: dict, workers: int = 5):
    """
    Executes all pipelines defined in the data dictionary.
    
    The 'data' dictionary keys ('classification', 'segmentation') are used
    to select the correct pipeline runner function.
    """
    pipeline_map = {
        'classification': run_classification_pipeline,
        'segmentation': run_segmentation_pipeline,
    }
    
    tasks = []
    
    for pipe_type, datasets in data.items():
        pipeline_func = pipeline_map.get(pipe_type)
        if not pipeline_func:
            print(f"Warning: Unknown pipeline type '{pipe_type}'. Skipping.")
            continue
            
        for dataset_name, path_list in datasets.items():
            print(f"Preparing tasks for {pipe_type} - {dataset_name}...")
            for raw_dir, out_dir in path_list:
                tasks.append((raw_dir, out_dir, pipeline_func))

    if not tasks:
        print("No tasks to run.")
        return

    print(f"\nExecuting {len(tasks)} total pipeline tasks with {workers} workers.")
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(task[2], task[0], task[1]) for task in tasks]
        
        for f in futures:
            f.result()
    print("\nAll tasks finished successfully.")

if __name__ == "__main__":
    data = {
        'classification': {
            'LocalDataSet': [
                ("./data/raw/LocalDataSet/DCL_Mammos", "./data/processed/LocalDataSet/DCL_Mammos"),
                ("./data/raw/LocalDataSet/DCL_USG", "./data/processed/LocalDataSet/DCL_USG"),
                ("./data/raw/LocalDataSet/Spectra_Mammos", "./data/processed/LocalDataSet/Spectra_Mammos")
            ],
        },
        'segmentation': {
            'BrCaWisconsin': [
                ("./data/raw/BrCaWisconsin", "./data/processed/BrCaWisconsin")
            ],
            'BrEaST-Lesions_USG-images_and_masks': [
                ("./data/raw/BrEaST-Lesions_USG-images_and_masks", "./data/processed/BrEaST-Lesions_USG-images_and_masks")
            ],
        },
    }

    run_all(data)