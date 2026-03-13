from scipy import stats
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

def clean_dice_scores(file):
    seg_metrics_save_dir = "./segmentation_metrics"
    os.makedirs(seg_metrics_save_dir, exist_ok=True)
    df = pd.read_csv(file)

    experiment_names = df['Experiment'].tolist()
    all_dice_scores = []

    for dice_score in df['Dice_Scores'].tolist():
        all_dice_scores.append([])
        dice_score = dice_score.strip("[]").split(",")
        for score in dice_score:
            clean_s = score.replace('np.float64(', '').replace(')', '')
            score = float(clean_s)
            all_dice_scores[-1].append(score)
    
    dice_dict = dict(zip(experiment_names, all_dice_scores))
    return dice_dict, seg_metrics_save_dir

dice_dict, seg_metrics_save_dir = clean_dice_scores("./first_eighteen_seg_results.csv")

experiment_pairs = list(combinations(dice_dict.keys(), 2))

comparison_results = []

experiment_names = list(dice_dict.keys())
n = len(experiment_names)

# Initialize a matrix for p-values (filled with 1.0 because comparing an exp to itself p=1)
p_matrix = pd.DataFrame(np.ones((n, n)), index=experiment_names, columns=experiment_names)
d_matrix = pd.DataFrame(np.ones((n, n)), index=experiment_names, columns=experiment_names)

def calculate_cohens_d(x1, x2):
    """Calculates Cohen's d for paired samples (repeated measures)."""
    diff = np.array(x1) - np.array(x2)
    # Using the standard deviation of the differences
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0
    return np.mean(diff) / std_diff

cohens_results = []
for exp1, exp2 in combinations(experiment_names, 2):
    d_val = calculate_cohens_d(dice_dict[exp1], dice_dict[exp2])
    
    d_matrix.loc[exp1, exp2] = d_val
    d_matrix.loc[exp2, exp1] = -d_val

    cohens_results.append({
        "Exp 1": exp1,
        "Exp 2": exp2,
        "Cohen's d": round(d_val, 4)
    })
    dice1 = dice_dict[exp1]
    dice2 = dice_dict[exp2]
    
    # Use Wilcoxon
    try:
        _, p_val = stats.wilcoxon(dice1, dice2)
    except (ValueError, TypeError):
        p_val = np.nan # Handle identical distributions or errors
        
    # Fill both sides of the matrix (A vs B is the same as B vs A)
    p_matrix.loc[exp1, exp2] = p_val # type: ignore
    p_matrix.loc[exp2, exp1] = p_val # type: ignore

pd.DataFrame(cohens_results).to_csv(os.path.join(seg_metrics_save_dir, "cohens_d_results.csv"), index=False)

# --- Plotting the P Heatmap ---
plt.figure(figsize=(14, 14))

# We use a Log-scale-like normalization or just highlight p < 0.05
# vmin/vmax set to 0 and 0.05 helps highlight the "significant" areas
sns.heatmap(p_matrix, annot=True, cmap="YlGnBu_r", vmin=0, vmax=0.05)

plt.title("P-Values Heatmap (Wilcoxon Signed-Rank Test)")
plt.xlabel("Experiment")
plt.ylabel("Experiment")
plt.tight_layout()

# Save the plot
plt.savefig(f"{seg_metrics_save_dir}/p_value_heatmap.png")
plt.show()
plt.close()

# --- Plotting the D Heatmap ---
plt.figure(figsize=(12, 10))

# We use a Log-scale-like normalization or just highlight p < 0.05
# vmin/vmax set to 0 and 0.05 helps highlight the "significant" areas
sns.heatmap(d_matrix, annot=True, cmap="RdBu_r", center=0, fmt=".2f")

plt.title("Cohen's d-Values Heatmap")
plt.xlabel("Experiment (Subtrahend)")
plt.ylabel("Experiment (Minuend)")
plt.tight_layout()

# Save the plot
plt.savefig(f"{seg_metrics_save_dir}/cohens_d_value_heatmap.png")
plt.show()
plt.close()