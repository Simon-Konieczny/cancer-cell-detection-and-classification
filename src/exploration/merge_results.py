import pandas as pd

results = ["./ablation_study_results.csv", "./class_mammo_ablation_study_results.csv"]

df = pd.DataFrame()
for r in results:
    df1 = pd.read_csv(r)
    df = pd.concat([df, df1], ignore_index=True)

df.to_csv("./class_mammo_ablation_study_results_combined.csv", index=False)