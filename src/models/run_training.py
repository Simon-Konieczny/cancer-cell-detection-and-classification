if __name__ == "__main__":
    input = input("classification or segmentation? (c/s): ").strip().lower()
    if input == "s":
        from train_segmentation import train_segmentation
        train_segmentation(
            processed_dir="./data/processed/LocalDataSet/DCL_USG",
            epochs=20,
            batch_size=16,
            lr=1e-3
        )
    else:
        from train_classification import train_classification
        train_classification(
            # processed_dir="./data/processed/LocalDataSet/DCL_USG",
            processed_dir="./data/processed/LocalDataSet/Spectra_Mammos",
            epochs=20,
            batch_size=32,
            lr=1e-3
        )