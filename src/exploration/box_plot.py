import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from stats_helper import clean_dice_scores

dice_dict, seg_metrics_save_dir = clean_dice_scores("./first_eighteen_seg_results.csv")

experiment_pairs = list(combinations(dice_dict.keys(), 2))

data = []
for exp, scores in dice_dict.items():
    for s in scores:
        data.append({"Experiment": exp, "Dice Score": s})

plot_df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))
sns.boxplot(x="Dice Score", y="Experiment", data=plot_df, palette="Set2")
sns.stripplot(x="Dice Score", y="Experiment", data=plot_df, color=".3", size=5) # Adds dots for the 5 folds

plt.title("Dice Score Distribution across 5 Folds")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{seg_metrics_save_dir}/dice_boxplot.png")
plt.show()
plt.close()