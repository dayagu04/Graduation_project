"""Regenerate Fig 3-5 (18-D Pearson heatmap) and Fig 4-3 (PSO-KMeans K=4 silhouette).

Reads pre-computed PSO-KMeans output to ensure figures match the thesis content.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "output" / "clustering_result_k4.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

FIG_HEATMAP = RESULTS / "1_特征工程_半三角相关性矩阵.png"
FIG_SILHOUETTE = RESULTS / "2_算法对比_K4轮廓系数分布图.png"

CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]


def plot_heatmap_18d() -> None:
    df = pd.read_csv(CSV).drop(columns=["cluster"])
    corr = df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(17, 15))
    sns.heatmap(
        corr,
        mask=~mask,  # show upper triangle only
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 14, "weight": "bold"},
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson 相关系数"},
        ax=ax,
    )
    ax.set_title(f"训练集 {df.shape[1]} 维联合特征空间 Pearson 相关性矩阵热力图",
                 fontsize=22, pad=18, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=17, fontweight="bold")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=17, fontweight="bold")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.label.set_size(17)
    plt.tight_layout()
    fig.savefig(FIG_HEATMAP, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[heatmap] dims={df.shape[1]} -> {FIG_HEATMAP}")


def plot_silhouette_pso_k4() -> None:
    df = pd.read_csv(CSV)
    labels = df["cluster"].to_numpy()
    X = df.drop(columns=["cluster"]).to_numpy()

    rng = np.random.default_rng(42)
    sample_size = min(5000, len(X))
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_s, lbl_s = X[idx], labels[idx]
    else:
        X_s, lbl_s = X, labels

    sil_vals = silhouette_samples(X_s, lbl_s)
    sil_avg = silhouette_score(X_s, lbl_s)

    cluster_ids = sorted(np.unique(lbl_s).tolist())
    n_clusters = len(cluster_ids)
    fig, ax = plt.subplots(figsize=(12, 8))

    y_lower = 10
    for i, ck in enumerate(cluster_ids):
        cluster_vals = np.sort(sil_vals[lbl_s == ck])
        size_i = cluster_vals.shape[0]
        y_upper = y_lower + size_i
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                         facecolor=color, edgecolor=color, alpha=0.78)
        cluster_mean = cluster_vals.mean()
        ax.text(
            -0.12, y_lower + size_i / 2,
            f"Cluster {ck}\n(n={size_i}, μ={cluster_mean:.3f})",
            ha="right", va="center", fontsize=14,
            color=color, fontweight="bold",
        )
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=2.2,
               label=f"平均轮廓系数 = {sil_avg:.4f}")
    ax.axvline(0, color="black", linewidth=1.2, alpha=0.5)

    ax.set_title(
        f"PSO-KMeans (k={n_clusters}) 各簇轮廓系数分布（采样 {sample_size} / {len(X)}）",
        fontsize=19, pad=16, fontweight="bold",
    )
    ax.set_xlabel("轮廓系数 (Silhouette Coefficient)", fontsize=17)
    ax.set_ylabel("样本（按簇分组并升序排列）", fontsize=17)

    x_min = min(-0.25, sil_vals.min() * 1.1)
    x_max = max(0.6, sil_vals.max() * 1.05)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_lower)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=15, framealpha=0.95)

    note = (
        "轮廓系数解释：\n"
        "· > 0：样本与本簇相似度高于其他簇\n"
        "· < 0：样本可能被错分到当前簇\n"
        "· 刀刃越长越厚 → 簇越紧密"
    )
    ax.text(
        0.98, 0.98, note, transform=ax.transAxes,
        fontsize=14, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f8f8",
                  edgecolor="gray", alpha=0.95, linewidth=1.2),
    )

    plt.tight_layout()
    fig.savefig(FIG_SILHOUETTE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[silhouette] n={len(X_s)} k={n_clusters} avg={sil_avg:.4f} -> {FIG_SILHOUETTE}")
    for i, ck in enumerate(cluster_ids):
        cs = sil_vals[lbl_s == ck]
        neg_pct = (cs < 0).mean() * 100
        print(f"  簇 {ck}: n={len(cs):>5d}  均值={cs.mean():+.4f}  "
              f"负样本占比={neg_pct:.2f}%")


if __name__ == "__main__":
    plot_heatmap_18d()
    plot_silhouette_pso_k4()
