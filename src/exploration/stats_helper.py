from scipy import stats
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

dice_dict, seg_metrics_save_dir = clean_dice_scores("./seg_results.csv")

experiment_pairs = list(combinations(dice_dict.keys(), 2))

comparison_results = []

experiment_names = list(dice_dict.keys())
n = len(experiment_names)

# Initialize a matrix for p-values (filled with 1.0 because comparing an exp to itself p=1)
p_matrix = pd.DataFrame(np.ones((n, n)), index=experiment_names, columns=experiment_names)
g_matrix = pd.DataFrame(np.ones((n, n)), index=experiment_names, columns=experiment_names)

def calculate_hedges_g(m1, sd1, m2, sd2, n=5):
    # Variance_sample = Variance_pop * (n / (n - 1))
    s1_sq = (sd1**2) * (n / (n - 1))
    s2_sq = (sd2**2) * (n / (n - 1))
    
    # Since n1 = n2, this is just the square root of the average variance
    s_pooled = np.sqrt((s1_sq + s2_sq) / 2)
    
    if s_pooled == 0:
        return 0.0
    
    d = (m1 - m2) / s_pooled
    
    df = (2 * n) - 2
    j = 1 - (3 / (4 * df - 1))
    
    return d * j

# def calculate_d_from_stats(m1, sd1, m2, sd2):
#     # m1, sd1 are the mean and std you saved
#     # Since you used np.std() (ddof=0) to save them, 
#     # we use them directly here.
    
#     pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
    
#     if pooled_sd == 0:
#         return 0
        
#     return (m1 - m2) / pooled_sd

hedges_results = []
for exp1, exp2 in combinations(experiment_names, 2):
    d1 = dice_dict[exp1]
    d2 = dice_dict[exp2]
    g_val = calculate_hedges_g(np.mean(d1), np.std(d1), np.mean(d2), np.std(d2))
    # d_val = calculate_cohens_d(dice_dict[exp1], dice_dict[exp2])
    
    g_matrix.loc[exp1, exp2] = g_val
    g_matrix.loc[exp2, exp1] = -g_val

    hedges_results.append({
        "Exp 1": exp1,
        "Exp 2": exp2,
        "Hedge's g": round(g_val, 4)
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

pd.DataFrame(hedges_results).to_csv(os.path.join(seg_metrics_save_dir, "hedges_g_results.csv"), index=False)

# --- Plotting the P Heatmap ---
fig1, ax1 = plt.subplots(figsize=(16,16))
# plt.figure(figsize=(14, 14))

sns.heatmap(p_matrix, annot=True, cmap="YlGnBu_r", vmin=0, vmax=0.05, ax=ax1)

ax1.set_title("P-Values Heatmap (Wilcoxon Signed-Rank Test)")
ax1.set_xlabel("Experiment")
ax1.set_ylabel("Experiment")
plt.tight_layout()

# Save the plot
plt.savefig(f"{seg_metrics_save_dir}/p_value_heatmap.png")
plt.show()
plt.close(fig1)

# --- Plotting the G Heatmap ---
fig2, ax2 = plt.subplots(figsize=(14,14))
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
    cbar_kws={'ticks': [-2.5, -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8, 2.5]},
    ax=ax2
)

ax2.set_title("Hedge's g-Values Heatmap")
ax2.set_xlabel("Experiment (Subtrahend)")
ax2.set_ylabel("Experiment (Minuend)")
plt.tight_layout()

plt.savefig(f"{seg_metrics_save_dir}/hedges_g_value_heatmap.png")
plt.show()
plt.close(fig2)