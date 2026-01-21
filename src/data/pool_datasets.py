import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def pool_datasets(dataset_dirs, output_base_dir):
    """
    dataset_dirs: List of paths to your processed dataset folders (e.g., ['./data/dataset1', './data/dataset2'])
    output_base_dir: Path where the pooled data will live (e.g., './data/processed/pooled')
    """
    pooled_path = Path(output_base_dir)
    pooled_images = pooled_path / "images"
    pooled_images.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []

    for d_path in dataset_dirs:
        d_path = Path(d_path)
        df = pd.read_csv(d_path / "metadata.csv")
        dataset_name = d_path.name
        
        print(f"Processing {dataset_name}...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            old_img_path = d_path / "images" / row['img_id']
            
            new_img_id = f"{dataset_name}_{row['img_id']}"
            new_img_path = pooled_images / new_img_id
            
            if old_img_path.exists():
                shutil.copy2(old_img_path, new_img_path)
                
                row_copy = row.copy()
                row_copy['img_id'] = new_img_id
                all_metadata.append(row_copy)
            else:
                print(f"Warning: Image {old_img_path} not found.")

    # Save the combined metadata
    final_df = pd.DataFrame(all_metadata)
    final_df.to_csv(pooled_path / "metadata.csv", index=False)
    print(f"\nSuccessfully pooled {len(final_df)} images into {pooled_path}")

if __name__ == "__main__":
    # datasets_to_pool = [
    #     './data/processed/LocalDataSet/DCL_Mammos',
    #     './data/processed/LocalDataSet/Spectra_Mammos'
    # ]
    # output_directory = './data/processed/pooled_Mammos'
    # pool_datasets(datasets_to_pool, output_directory)

    datasets_to_pool = [
        './data/processed/BrCaWisconsin',
        './data/processed/LocalDataSet/DCL_USG',
    ]
    output_directory = './data/processed/pooled_USG'
    pool_datasets(datasets_to_pool, output_directory)