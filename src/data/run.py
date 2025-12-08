from extract import extract_classification, extract_segmentation
from clean import remove_duplicates
from standardise import apply_standardisation
from concurrent.futures import ProcessPoolExecutor
from split import split_cla_dataset, split_seg_dataset
import yaml
from pathlib import Path

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
            for path_config in path_list:
                raw_dir = path_config['raw']
                out_dir = path_config['processed']
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
    SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = SCRIPT_DIR / 'config.yaml'
    print(f"Loading configuration from: {CONFIG_PATH}")

    try:
        if not CONFIG_PATH.exists():
             raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        pipeline_data = config.get('pipeline_data', {})
        
        run_all(pipeline_data)

    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_PATH}' not found. Exiting.")
    except ImportError:
        print("Error: PyYAML library not installed. Please run 'pip install pyyaml'.")
    except Exception as e:
        print(f"An unexpected error occurred during configuration loading: {e}")