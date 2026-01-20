import pandas as pd

df = pd.read_csv("./data/processed/BrCaWisconsin/metadata.csv")
print(df["label"].value_counts())
print(f'Total samples: {len(df)}')