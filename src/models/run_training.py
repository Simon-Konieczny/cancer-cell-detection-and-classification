if __name__ == "__main__":
    input = input("classification or segmentation? (c/s): ").strip().lower()
    if input == "s":
        from train_segmentation import segmentation
        segmentation(
            processed_dir="./data/processed/pooled_seg_USG",
            epochs=100,
            batch_size=8
        )
    else:
        from train_classification import classification
        classification(
            # processed_dir="./data/processed/pooled_USG",
            processed_dir="./data/processed/pooled_Mammos",
            epochs=100,
            batch_size=12,
        )