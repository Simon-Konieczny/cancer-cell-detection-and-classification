import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import ast

from stats_helper import clean_dice_scores

def get_box_plt(hd=False):
    if hd:
        h_df = pd.read_csv("./segmentation_metrics/hd95_results.csv", 
                   converters={'HD95s': ast.literal_eval})
        plot_df = h_df.explode('HD95s')
        
        
        fig1, ax1 = plt.subplots(figsize=(12,6))
        sns.boxplot(x="HD95s", y="Experiment", data=plot_df, palette="Set2", ax=ax1)
        sns.stripplot(x="HD95s", y="Experiment", data=plot_df, color=".3", size=5, ax=ax1)

        ax1.set_title("HD95 Metric Distibution across 5 Folds")
        ax1.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        fig1.savefig(f"./segmentation_metrics/hd95_boxplot.png")
        plt.close(fig1)
    else:
        dice_dict, seg_metrics_save_dir = clean_dice_scores("./seg_results.csv")

        experiment_pairs = list(combinations(dice_dict.keys(), 2))

        data = []
        for exp, scores in dice_dict.items():
            for s in scores:
                data.append({"Experiment": exp, "Dice Score": s})

        plot_df = pd.DataFrame(data)
        print(plot_df)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Dice Score", y="Experiment", data=plot_df, palette="Set2")
        sns.stripplot(x="Dice Score", y="Experiment", data=plot_df, color=".3", size=5) # Adds dots for the 5 folds

        plt.title("Dice Score Distribution across 5 Folds")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{seg_metrics_save_dir}/dice_boxplot.png")
        plt.show()
        plt.close()

get_box_plt(hd=True)