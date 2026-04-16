import pandas as pd

# dcl_mammos_df = pd.read_csv("./data/processed/LocalDataSet/DCL_Mammos/metadata.csv")
# spectra_mammos_df = pd.read_csv("./data/processed/LocalDataSet/Spectra_Mammos/metadata.csv")
# wisconcin_df = pd.read_csv("./data/processed/BrCaWisconsin/metadata.csv")
# dcl_usg_df = pd.read_csv("./data/processed/LocalDataSet/DCL_USG/metadata.csv")
# breast_df = pd.read_csv("./data/processed/BrEaST-Lesions_USG-images_and_masks/metadata.csv")

# print(f"pooled mammos counts: dcl_mammos: {dcl_mammos_df.count()} spectra: {spectra_mammos_df.count()}")
# print(f"pooled usg counts: braCaWis: {wisconcin_df.count()} spectra: {dcl_usg_df.count()}")
# print(f"pooled usg seg counts: braCaWis: {wisconcin_df.count()} spectra: {breast_df.count()}")

pooled_mammos = pd.read_csv('./data/processed/pooled_Mammos/metadata.csv')
pooled_seg = pd.read_csv('./data/processed/pooled_seg_USG/metadata.csv')
pooled_usg = pd.read_csv('./data/processed/pooled_USG/metadata.csv')

for df in [pooled_mammos, pooled_seg, pooled_usg]:
    print("-"*30)
    print(f"Benign: {len(df[df['label'] == 'benign'])}")
    print(f"Malignant: {len(df[df['label'] == 'malignant'])}")
    print(f"Normal: {len(df[df['label'] == 'normal'])}")