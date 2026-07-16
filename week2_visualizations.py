"""
Week 2 - Advanced Data Visualization and Storytelling
Dataset: breast_cancer_clean.csv (Wisconsin Diagnostic Breast Cancer, cleaned in Week 1)

Run:  python week2_visualizations.py
Produces: figures/fig1..fig6 (PNG) and figures/stats.json
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "breast_cancer_clean.csv"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Style: one consistent, colour-blind-friendly story palette ----------
BEN, MAL = "#2a9d8f", "#e76f51"          # teal = benign, coral = malignant
PAL = {"Benign": BEN, "Malignant": MAL}
INK, GREY = "#264653", "#8d99ae"
sns.set_theme(style="whitegrid", context="notebook", font="DejaVu Sans")
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "grid.color": "#e5e8ec",
})

df = pd.read_csv(DATA)
orig_cols = [c for c in df.columns if c != "diagnosis" and not c.endswith("_yj")]
n = len(df)
counts = df["diagnosis"].value_counts()
ben, mal = df[df.diagnosis == "Benign"], df[df.diagnosis == "Malignant"]
S = {"n_rows": n, "n_features": len(orig_cols),
     "benign": int(counts["Benign"]), "malignant": int(counts["Malignant"])}


def nice(c):
    return c.replace("_", " ").replace("fractal dimension", "fractal dim.").capitalize()


# =====================================================================
# FIG 1 - The cast: class balance + the first clue (tumour size)
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})
wedges, _ = ax[0].pie([counts["Benign"], counts["Malignant"]], colors=[BEN, MAL],
                      startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.38, edgecolor="white", linewidth=3))
ax[0].text(0, 0.08, f"{n}", ha="center", fontsize=26, fontweight="bold", color=INK)
ax[0].text(0, -0.18, "tumour samples", ha="center", fontsize=10, color=GREY)
ax[0].legend(wedges, [f"Benign  {counts['Benign']} ({counts['Benign']/n:.0%})",
                      f"Malignant  {counts['Malignant']} ({counts['Malignant']/n:.0%})"],
             loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=1)
ax[0].set_title("Who is in the data?", loc="left")

for d, c in PAL.items():
    sns.kdeplot(df.loc[df.diagnosis == d, "radius_mean"], ax=ax[1], color=c, fill=True,
                alpha=.35, linewidth=2, label=d)
    ax[1].axvline(df.loc[df.diagnosis == d, "radius_mean"].median(), color=c, ls="--", lw=1.4)
mb, mm = ben.radius_mean.median(), mal.radius_mean.median()
ax[1].set_ylim(0, 0.28); ax[1].set_xlim(0, 33)
ax[1].text(mb - 0.4, 0.255, f"Benign median {mb:.1f}", color=BEN, ha="right", fontsize=10, fontweight="bold")
ax[1].text(mm + 0.4, 0.255, f"Malignant median {mm:.1f}", color=MAL, ha="left", fontsize=10, fontweight="bold")
ax[1].set_xlabel("Mean cell-nucleus radius (lab units)")
ax[1].set_ylabel("Density")
ax[1].set_title("Malignant nuclei are visibly larger", loc="left")
ax[1].legend(frameon=False, loc="center right")
fig.suptitle("Chapter 1 - Meeting the data", x=0.02, ha="left", fontsize=15,
             fontweight="bold", color=INK, y=1.03)
fig.savefig(f"{OUT}/fig1_overview.png"); plt.close(fig)
S["radius_median_benign"], S["radius_median_malignant"] = round(mb, 2), round(mm, 2)

# =====================================================================
# FIG 2 - Four "alarm" features: violin + points, with outliers named
# =====================================================================
feats = ["area_worst", "perimeter_worst", "concave_points_worst", "texture_mean"]
titles = ["Worst area", "Worst perimeter", "Worst concave points", "Mean texture"]
fig, axes = plt.subplots(1, 4, figsize=(13, 4.6))
for a, f, t in zip(axes, feats, titles):
    sns.violinplot(data=df, x="diagnosis", y=f, hue="diagnosis", palette=PAL, inner=None,
                   cut=0, linewidth=0, alpha=.55, ax=a, legend=False,
                   order=["Benign", "Malignant"])
    sns.boxplot(data=df, x="diagnosis", y=f, width=.14, showfliers=False, boxprops=dict(
        facecolor="white", edgecolor=INK), whiskerprops=dict(color=INK), capprops=dict(color=INK),
        medianprops=dict(color=INK, lw=2), ax=a, order=["Benign", "Malignant"])
    sns.stripplot(data=df, x="diagnosis", y=f, hue="diagnosis", palette=PAL, size=1.8,
                  alpha=.35, jitter=.25, ax=a, legend=False, order=["Benign", "Malignant"])
    d = (mal[f].mean() - ben[f].mean()) / np.sqrt((mal[f].var() + ben[f].var()) / 2)
    a.set_title(f"{t}  (d = {d:.1f})", loc="left", fontsize=12)
    a.set_xlabel(""); a.set_ylabel(""); a.margins(y=0.08)
# call out the single most extreme sample in the area panel
i = df["area_worst"].idxmax()
axes[0].annotate("Most extreme\ntumour in dataset", (1, df.loc[i, "area_worst"]),
                 xytext=(-0.45, df.loc[i, "area_worst"] * 0.93), fontsize=8.5, color=MAL,
                 arrowprops=dict(arrowstyle="->", color=MAL))
fig.suptitle("Chapter 2 - Where do the groups pull apart? (box = middle 50 %, line = median; d = effect size)",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_distributions.png"); plt.close(fig)
S["area_worst_median_benign"] = round(ben.area_worst.median(), 0)
S["area_worst_median_malignant"] = round(mal.area_worst.median(), 0)
S["texture_d"] = round((mal.texture_mean.mean() - ben.texture_mean.mean()) /
                       np.sqrt((mal.texture_mean.var() + ben.texture_mean.var()) / 2), 2)

# =====================================================================
# FIG 3 - Redundancy: clustered correlation heatmap of the 10 "mean" features
# =====================================================================
mean_cols = [c for c in orig_cols if c.endswith("_mean")]
corr = df[mean_cols].corr()
order = sns.clustermap(corr, method="average").dendrogram_row.reordered_ind
plt.close("all")
corr = corr.iloc[order, order]
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(8.6, 7))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 20, as_cmap=True), vmin=-1,
            vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=1,
            linecolor="white", square=True, cbar_kws={"shrink": .7, "label": "Correlation"}, ax=ax)
ax.set_xticklabels([nice(c.replace("_mean", "")) for c in corr.columns], rotation=45, ha="right")
ax.set_yticklabels([nice(c.replace("_mean", "")) for c in corr.index], rotation=0)
ax.set_title("Chapter 3 - Many measurements, few independent ideas", loc="left",
             fontsize=14, color=INK)
rp = corr.loc["radius_mean", "perimeter_mean"]; ra = corr.loc["radius_mean", "area_mean"]
ax.grid(False)
ax.text(0.5, 0.80, f"Radius, perimeter and area are\nnear-identical measurements\n(r = {rp:.3f} and {ra:.3f})",
        transform=ax.transAxes, fontsize=11, color=MAL, va="center", fontweight="bold")
fig.savefig(f"{OUT}/fig3_correlation.png"); plt.close(fig)
S["corr_radius_perimeter"], S["corr_radius_area"] = round(rp, 3), round(ra, 3)
hi = (df[orig_cols].corr().abs().where(np.triu(np.ones((len(orig_cols),) * 2), 1).astype(bool)) > .9).sum().sum()
S["pairs_above_0.9"] = int(hi)

# =====================================================================
# FIG 4 - The decision map: two features, marginal densities, overlap zone
# =====================================================================
g = sns.JointGrid(data=df, x="radius_worst", y="concave_points_worst", hue="diagnosis",
                  palette=PAL, height=7, ratio=5, space=0.1)
g.plot_joint(sns.scatterplot, s=34, alpha=.7, edgecolor="white", linewidth=.4)
g.plot_marginals(sns.kdeplot, fill=True, alpha=.4, linewidth=1.6)
ax = g.ax_joint
ax.set_xlabel("Worst radius (largest nucleus in sample)")
ax.set_ylabel("Worst concave points (how many dents in the outline)")
# simple, explainable rule boundary from a 2-feature logistic model
X2 = df[["radius_worst", "concave_points_worst"]]
y2 = (df.diagnosis == "Malignant").astype(int)
m2 = make_pipeline(StandardScaler(), LogisticRegression()).fit(X2, y2)
xx, yy = np.meshgrid(np.linspace(X2.iloc[:, 0].min(), X2.iloc[:, 0].max(), 300),
                     np.linspace(X2.iloc[:, 1].min(), X2.iloc[:, 1].max(), 300))
zz = m2.predict_proba(pd.DataFrame({"radius_worst": xx.ravel(),
                                    "concave_points_worst": yy.ravel()}))[:, 1].reshape(xx.shape)
ax.contour(xx, yy, zz, levels=[.5], colors=INK, linewidths=2, linestyles="--")
acc2 = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X2, y2, cv=5).mean()
ax.set_xlim(6, 38)
ax.text(0.012, 0.62, "Benign side\nof the line", transform=ax.transAxes, color=BEN, fontweight="bold", fontsize=11)
ax.text(0.97, 0.80, "Malignant side\nof the line", transform=ax.transAxes, color=MAL, fontweight="bold", ha="right", fontsize=11)
ax.annotate("Grey zone: a few cases of\nboth types sit near the line", xy=(17.2, 0.115), xytext=(25, 0.03),
            fontsize=10, color=INK, ha="left",
            arrowprops=dict(arrowstyle="->", color=INK),
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec=GREY, alpha=.95))
ax.legend(title="", frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
g.fig.suptitle("Chapter 4 - Two measurements already separate most cases (dashed line = 50/50)",
               x=0.02, ha="left", y=1.02, fontsize=13, fontweight="bold", color=INK)
g.fig.savefig(f"{OUT}/fig4_decision_map.png"); plt.close(g.fig)
S["two_feature_cv_accuracy"] = round(float(acc2), 3)

# =====================================================================
# FIG 5 - Ranking every feature by how strongly it separates the groups
# =====================================================================
rows = []
for c in orig_cols:
    d = (mal[c].mean() - ben[c].mean()) / np.sqrt((mal[c].var() + ben[c].var()) / 2)
    rows.append((c, d))
eff = pd.DataFrame(rows, columns=["feature", "d"]).sort_values("d")
eff["group"] = eff.feature.str.extract(r"_(mean|se|worst)$")[0]
gpal = {"mean": "#457b9d", "se": "#b8c0cc", "worst": MAL}
fig, ax = plt.subplots(figsize=(8.6, 9))
y = np.arange(len(eff))
ax.hlines(y, 0, eff.d, color=eff.group.map(gpal), lw=2.2)
ax.scatter(eff.d, y, color=eff.group.map(gpal), s=60, zorder=3, edgecolor="white")
ax.set_yticks(y); ax.set_yticklabels([nice(f) for f in eff.feature], fontsize=8.5)
ax.axvline(0, color=INK, lw=1)
ax.axvline(0.8, color=GREY, ls=":", lw=1.2)
ax.text(0.82, 1, "'large effect' (d > 0.8)", color=GREY, fontsize=9)
ax.set_xlabel("Effect size (Cohen's d): how far apart the two group averages are")
ax.set_title("Chapter 5 - Which measurements matter most?", loc="left", fontsize=14, color=INK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=gpal[k], lw=3, label=l) for k, l in
                   [("worst", "Worst (most extreme cell)"), ("mean", "Mean (average cell)"),
                    ("se", "Std. error (variability)")]], frameon=False, loc="lower right")
fig.savefig(f"{OUT}/fig5_effect_sizes.png"); plt.close(fig)
top = eff.sort_values("d", ascending=False).head(3)
S["top3_features"] = [(r.feature, round(r.d, 2)) for r in top.itertuples()]
S["n_large_effect"] = int((eff.d > 0.8).sum())
S["mean_d_by_group"] = {k: round(v, 2) for k, v in eff.groupby("group").d.mean().items()}
S["weakest"] = [(r.feature, round(r.d, 2)) for r in eff.reindex(eff.d.abs().sort_values().index).head(3).itertuples()]

# =====================================================================
# FIG 6 - The big picture: 30 measurements compressed into 2 dimensions (PCA)
# =====================================================================
Xs = StandardScaler().fit_transform(df[orig_cols])
pca = PCA().fit(Xs)
Z = pca.transform(Xs)[:, :2]
ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.7, 1]})
for d, c in PAL.items():
    mk = (df.diagnosis == d).values
    ax[0].scatter(Z[mk, 0], Z[mk, 1], c=c, s=28, alpha=.7, edgecolor="white", linewidth=.4, label=d)
    ell_c = Z[mk].mean(0)
    ax[0].scatter(*ell_c, marker="X", s=220, c=c, edgecolor=INK, linewidth=1.4, zorder=5)
ax[0].set_xlabel(f"Principal component 1 ({ev[0]:.0%} of all variation)")
ax[0].set_ylabel(f"Principal component 2 ({ev[1]:.0%})")
ax[0].legend(frameon=False, title="X = group average")
ax[0].set_title("All 30 measurements, flattened to a single map", loc="left")
cum = np.cumsum(ev)
ax[1].bar(range(1, 11), ev[:10], color=GREY, alpha=.6, label="Per component")
ax[1].plot(range(1, 11), cum[:10], "-o", color=INK, label="Cumulative")
ax[1].axhline(.9, color=MAL, ls="--", lw=1)
k90 = int(np.argmax(cum >= .9)) + 1
ax[1].annotate(f"{k90} components\nkeep 90 %", (k90, cum[k90 - 1]), xytext=(k90 + 1.5, .62),
               arrowprops=dict(arrowstyle="->", color=MAL), color=MAL)
ax[1].set_xlabel("Component"); ax[1].set_ylabel("Share of variation")
ax[1].set_xticks(range(1, 11)); ax[1].set_ylim(0, 1.05)
ax[1].legend(frameon=False, loc="center right")
ax[1].set_title("How much do we lose?", loc="left")
fig.suptitle("Chapter 6 - The whole picture: benign and malignant cases settle into largely separate regions",
             x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_pca.png"); plt.close(fig)
S["pc1_var"], S["pc2_var"] = round(ev[0], 3), round(ev[1], 3)
S["components_for_90pct"] = k90

# ---------- A quantified "so what": full-feature model, cross-validated ----------
yfull = (df.diagnosis == "Malignant").astype(int)
acc_full = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                           df[orig_cols], yfull, cv=5).mean()
S["all_feature_cv_accuracy"] = round(float(acc_full), 3)
S["pc12_cv_accuracy"] = round(float(cross_val_score(
    LogisticRegression(), Z, yfull, cv=5).mean()), 3)

json.dump(S, open(f"{OUT}/stats.json", "w"), indent=2, default=str)
print(json.dumps(S, indent=2, default=str))
