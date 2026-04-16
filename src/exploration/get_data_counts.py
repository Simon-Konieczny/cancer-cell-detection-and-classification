import pandas as pd

pooled_mammos = pd.read_csv('./data/processed/pooled_Mammos/metadata.csv')
pooled_seg = pd.read_csv('./data/processed/pooled_seg_USG/metadata.csv')
pooled_usg = pd.read_csv('./data/processed/pooled_USG/metadata.csv')

for df in [pooled_mammos, pooled_seg, pooled_usg]:
    print("-"*30)
    print(f"Benign: {len(df[df['label'] == 'benign'])}")
    print(f"Malignant: {len(df[df['label'] == 'malignant'])}")
    print(f"Normal: {len(df[df['label'] == 'normal'])}")