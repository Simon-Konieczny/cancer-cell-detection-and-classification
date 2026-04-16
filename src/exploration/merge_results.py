import pandas as pd

results = ["./no_aug_res.csv", "./seg_results_old.csv"]

df = pd.DataFrame()
for r in results:
    df1 = pd.read_csv(r)
    df = pd.concat([df, df1], ignore_index=True)

df.to_csv("./seg_results.csv", index=False)