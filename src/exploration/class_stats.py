import numpy as np
import pandas as pd
from itertools import combinations
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from stats_helper import calculate_hedges_g

class_metrics_save_dir = "./classification_metrics"
bs = "usg"
os.makedirs(class_metrics_save_dir, exist_ok=True)
df = pd.read_csv(f"./class_{bs}_ablation_study_results.csv")
exps = df["Experiment"].tolist()
f1_means = df["F1_Mean"].tolist()
f1_stds = df["F1_Std"].tolist()
means_dict = dict(zip(exps, f1_means))
stds_dict = dict(zip(exps, f1_stds))

experiment_pairs = list(combinations(exps, 2))
n = len(exps)
g_matrix = pd.DataFrame(np.ones((n, n)), index=exps, columns=exps)

hedges_results = []
for exp1, exp2 in experiment_pairs:
    g_val = calculate_hedges_g(means_dict[exp1], stds_dict[exp1], means_dict[exp2], stds_dict[exp2])

    g_matrix.loc[exp1, exp2] = g_val
    g_matrix.loc[exp2, exp1] = -g_val

    hedges_results.append({
        "Exp 1": exp1,
        "Exp 2": exp2,
        "Hedge's g": round(g_val, 4)
    })

pd.DataFrame(hedges_results).to_csv(os.path.join(class_metrics_save_dir, f"{bs}_hedges_g_results.csv"), index=False)

fig1, ax1 = plt.subplots(figsize=(12,10))
# plt.figure(figsize=(12, 10))

data_min = min(-1.0, np.min(g_matrix))
data_max = max(1.0, np.max(g_matrix))
boundaries = [data_min, -2.5, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 2.5, data_max]
colors = [
    "#67001f",
    "#b2182b",
    "#ef8a62",
    "#fddbc7",
    "#f7f7f7",
    "#d1e5f0",
    "#67a9cf",
    "#2166ac",
    "#053061"
]

custom_cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(boundaries, custom_cmap.N)

sns.heatmap(
    g_matrix, 
    annot=True, 
    cmap=custom_cmap, 
    norm=norm, 
    fmt=".2f",
    cbar_kws={'ticks': [-0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8]},
    ax=ax1
)

ax1.set_title("Hedge's g-Values Heatmap")
ax1.set_xlabel("Experiment (Subtrahend)")
ax1.set_ylabel("Experiment (Minuend)")
plt.tight_layout()

plt.savefig(f"{class_metrics_save_dir}/{bs}_hedges_g_value_heatmap.png")
plt.show()
plt.close(fig1)