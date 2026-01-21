import pandas as pd

df = pd.read_csv("./data/processed/pooled_USG/metadata.csv")
print('usg')
print(df["label"].value_counts())
print(f'Total samples: {len(df)}')

df = pd.read_csv("./data/processed/pooled_Mammos/metadata.csv")
print('mammos')
print(df["label"].value_counts())
print(f'Total samples: {len(df)}')

# df = pd.read_csv("./data/processed/LocalDataSet/DCL_USG/metadata.csv")
# print('usg')
# print(df["label"].value_counts())
# print(f'Total samples: {len(df)}')
# print(f'Missing labels: {df["label"].isnull().sum()}')

# df = pd.read_csv("./data/processed/BrEaST-Lesions_USG-images_and_masks/metadata.csv")
# print('breast_lesions_usg')
# print(df["label"].value_counts())
# print(f'Total samples: {len(df)}')
# print(f'Missing labels: {df["label"].isnull().sum()}')

# df = pd.read_csv("./data/processed/BrCaWisconsin/metadata.csv")
# print('wisconsin')
# print(df["label"].value_counts())
# print(f'Total samples: {len(df)}')
# print(f'Missing labels: {df["label"].isnull().sum()}')