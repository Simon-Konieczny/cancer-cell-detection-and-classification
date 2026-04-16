import pandas as pd
import numpy as np
import ast

s_df = pd.read_csv("./seg_results.csv")

all_precision_scores = []
all_precision_std = []

all_recall_scores = []
all_recall_std = []

for p_score in s_df['Precision_Scores'].tolist():
        all_precision_scores.append([])
        p_score = p_score.strip("[]").split(",")
        for score in p_score:
            clean_s = score.replace('np.float64(', '').replace(')', '')
            score = float(clean_s)
            all_precision_scores[-1].append(score)

for r_score in s_df['Recall_Scores'].tolist():
        all_recall_scores.append([])
        r_score = r_score.strip("[]").split(",")
        for score in r_score:
            clean_s = score.replace('np.float64(', '').replace(')', '')
            score = float(clean_s)
            all_recall_scores[-1].append(score)

for s in all_precision_scores:
      all_precision_std.append(np.std(s))

for r in all_recall_scores:
      all_recall_std.append(np.std(r))


s_df["Precision_Std"] = all_precision_std
s_df["Recall_Std"] = all_recall_std

s_df.drop(["Dice_Scores", "Precision_Scores", "Recall_Scores"], axis=1, inplace=True)

# H Score
h_df = pd.read_csv("./segmentation_metrics/hd95_results_new.csv", 
                   converters={'HD95s': ast.literal_eval})

all_hdfs, all_hdf_stds = [], []
for exp in h_df["HD95s"]:
    all_hdfs.append(np.mean(exp))
    all_hdf_stds.append(np.std(exp))

h_df.drop("HD95s", axis=1, inplace=True)
h_df["HD95_Mean"] = all_hdfs
h_df["HD95_Std"] = all_hdf_stds

res_df = s_df.merge(h_df, how="outer", on="Experiment")
res_df.to_csv("./segmentation_metrics/full_metrics_new.csv", index=False)