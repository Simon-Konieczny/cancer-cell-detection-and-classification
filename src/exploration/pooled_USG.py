import pandas as pd

df = pd.read_csv("./data/processed/pooled_USG/metadata.csv")
print('usg')
print(df["label"].value_counts())
print(f'Total samples: {len(df)}')
print(df["split"].value_counts())