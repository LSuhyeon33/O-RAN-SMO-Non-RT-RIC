"""
Per-class f1_score Heatmap for IDS rApp AI/ML model comparison.

Usage:
    python plot_heatmap.py

Outputs:
    per_class_accuracy_heatmap.png  (300 dpi, publication-ready)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------
# 1. Data — replace with your own values if needed
# ---------------------------------------------------------------
models  = ["XGBoost", "Random Forest", "Bi-LSTM", "LSTM", "TCN", "Transformer"]
classes = ["normal", "S1-normal", "S1-termination", "S2", "S3", "S4"]

# rows = models, columns = classes
accuracy = np.array([
    # normal, S1-normal, S1-term, S2,    S3,    S4
    [1.00,    0.03,      1.00,    0.45,  0.21,  0.94],   # XGBoost
    [1.00,    0.03,      1.00,    0.43,  0.11,  0.92],   # Random Forest
    [1.00,    0.43,      1.00,    0.78,  1.00,  0.96],   # Bi-LSTM
    [1.00,    0.03,      1.00,    0.44,  0.94,  0.94],   # LSTM
    [1.00,    0.03,      1.00,    0.45,  1.00,  0.93],   # TCN
    [1.00,    0.01,      1.00,    0.43,  0.84,  0.92],   # Transformer
])
f1_score = np.array([
    # normal, S1-normal, S1-term, S2,    S3,    S4
    [1.00,    0.05,      0.56,    0.62,  0.29,  0.92],   # XGBoost
    [0.94,    0.05,      0.53,    0.60,  0.17,  0.93],   # Random Forest
    [1.00,    0.59,      0.82,    0.87,  0.96,  0.86],   # Bi-LSTM
    [0.94,    0.05,      0.70,    0.61,  0.88,  0.80],   # LSTM
    [0.95,    0.05,      0.72,    0.62,  0.91,  0.84],   # TCN
    [0.94,    0.02,      0.63,    0.60,  0.86,  0.91],   # Transformer
])

df = pd.DataFrame(f1_score, index=models, columns=classes)

# Optional: row-wise mean (model average across classes) as an extra column
df["Avg."] = df.mean(axis=1).round(3)


# ---------------------------------------------------------------
# 2. Style — academic look (Times-like serif, tight layout)
# ---------------------------------------------------------------
plt.rcParams.update({
    # "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
})


# ---------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6))

# Separator between f1_score columns and the average column
sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0.0, vmax=1.0,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "f1_score", "shrink": 0.85, "pad": 0.02},
    annot_kws={"size": 10, "weight": "medium"},
    ax=ax,
)

# Draw a thin divider before the Avg. column to mark it as a summary
ax.axvline(x=len(classes), color="black", linewidth=1.2)

ax.set_xlabel("Class", labelpad=8)
ax.set_ylabel("Model", labelpad=8)
ax.set_title("Per-class f1_score Comparison", pad=10)

# Rotate class labels for readability
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=0)

plt.tight_layout()

out_path = "per_class_f1_score_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

# Uncomment to display interactively:
# plt.show()
