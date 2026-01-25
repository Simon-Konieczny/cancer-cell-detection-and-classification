import pandas as pd

df = pd.read_csv("./data/processed/pooled_seg_USG/metadata.csv")
print('seg usg')
print(f'Total samples: {len(df)}')
print(f'validation set: {len(df[df["split"]=="val"])}')
print(f'train set: {len(df[df["split"]=="train"])}')