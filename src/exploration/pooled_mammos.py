import pandas as pd

df = pd.read_csv("./data/processed/pooled_Mammos/metadata.csv")
print('mammo')
print(df["label"].value_counts())
print(f'Total samples: {len(df)}')