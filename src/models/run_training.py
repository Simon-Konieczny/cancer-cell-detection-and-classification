from train import train_model

if __name__ == "__main__":
    train_model(
        processed_dir="./data/processed/LocalDataSet/DCL_Mammos",
        epochs=20,
        batch_size=32,
        lr=1e-3
    )