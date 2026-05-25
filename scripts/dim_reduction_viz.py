"""
补齐 3 张论文图：
  1. 1_特征工程_特征对比.png    —— 原始特征空间 vs 衍生特征空间对比
  2. 3_聚类结果_PCA降维.png      —— PCA 线性映射 k=4 聚类二维散点
  3. 3_聚类结果_tSNE降维.png     —— t-SNE 非线性流形 k=4 聚类二维散点
所有图统一 600 DPI、加大字号，适配 Word 插入。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, prepare
from pso_kmeans import PSOKMeans

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
CLUSTER_NAMES = {
    0: "簇 0",
    1: "簇 1",
    2: "簇 2",
    3: "簇 3",
}


# ── 图 1：原始特征 vs 衍生特征空间对比 ────────────────────────────────────────

def plot_feature_comparison(out_path: Path) -> None:
    X, names, _ = prepare(TRAIN_CONFIG)
    N = len(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) 原始特征空间：StudyHours × ExamScore
    x1, y1 = X["StudyHours"].values, X["ExamScore"].values
    r1 = np.corrcoef(x1, y1)[0, 1]
    ax = axes[0]
    ax.scatter(x1, y1, alpha=0.25, s=14, color="#1f77b4", edgecolors="none")
    ax.set_xlabel("StudyHours", fontsize=16, fontweight="bold")
    ax.set_ylabel("ExamScore", fontsize=16, fontweight="bold")
    ax.set_title("(a) 原始特征空间",
                 fontsize=17, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    ax.text(0.03, 0.97,
            f"Pearson r = {r1:+.3f}\nN = {N}",
            transform=ax.transAxes, fontsize=14,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    # (b) 衍生特征空间：study_efficiency × academic_composite
    x2 = X["study_efficiency"].values
    y2 = X["academic_composite"].values
    r2 = np.corrcoef(x2, y2)[0, 1]
    ax = axes[1]
    ax.scatter(x2, y2, alpha=0.25, s=14, color="#d62728", edgecolors="none")
    ax.set_xlabel("study_efficiency", fontsize=16, fontweight="bold")
    ax.set_ylabel("academic_composite", fontsize=16, fontweight="bold")
    ax.set_title("(b) 衍生特征空间",
                 fontsize=17, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    ax.text(0.03, 0.97,
            f"Pearson r = {r2:+.3f}\nN = {N}",
            transform=ax.transAxes, fontsize=14,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    plt.suptitle("原始特征与衍生特征空间对比分析",
                 fontsize=20, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  原始空间 r={r1:.4f}  衍生空间 r={r2:.4f}")


# ── 图 2：PCA 线性映射聚类二维散点 ────────────────────────────────────────────

def plot_pca_scatter(out_path: Path) -> None:
    df = pd.read_csv(Path("output") / "clustering_result_k4.csv")
    labels = df["cluster"].to_numpy()
    X_vals = df.drop(columns=["cluster"]).to_numpy()

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_vals)
    centers = np.vstack([X_vals[labels == i].mean(axis=0) for i in range(4)])
    centers_pca = pca.transform(centers)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i in range(4):
        mask = labels == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=CLUSTER_COLORS[i], alpha=0.45, s=18,
                   edgecolors="none",
                   label=f"{CLUSTER_NAMES[i]}  (n={int(mask.sum())})")

    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
               c="black", marker="X", s=320,
               edgecolors="white", linewidth=2.2,
               label="聚类中心", zorder=10)

    pc1, pc2 = pca.explained_variance_ratio_
    ax.set_xlabel(f"主成分 1 (PC1, {pc1:.2%} 方差)", fontsize=15)
    ax.set_ylabel(f"主成分 2 (PC2, {pc2:.2%} 方差)", fontsize=15)
    ax.set_title("基于 PCA 线性映射的聚类结果二维分布散点图（k=4）",
                 fontsize=17, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=12, framealpha=0.93,
              markerscale=1.2)

    info = (f"累计解释方差 = {(pc1 + pc2):.2%}\n"
            f"样本总数 N = {len(X_vals)}\n"
            f"聚类算法 = PSO-KMeans")
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            fontsize=12, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor="gray", alpha=0.93))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  PC1 方差占比 {pc1:.4f}  PC2 方差占比 {pc2:.4f}  累计 {pc1+pc2:.4f}")


# ── 图 3：t-SNE 非线性流形聚类二维散点 ────────────────────────────────────────

def plot_tsne_scatter(out_path: Path, sample_size: int = 4000) -> None:
    df = pd.read_csv(Path("output") / "clustering_result_k4.csv")
    labels = df["cluster"].to_numpy()
    X_vals = df.drop(columns=["cluster"]).to_numpy()

    rng = np.random.default_rng(42)
    sample_size = min(sample_size, len(X_vals))
    idx = rng.choice(len(X_vals), size=sample_size, replace=False)
    X_sub, y_sub = X_vals[idx], labels[idx]

    print(f"  t-SNE 输入：{X_sub.shape[0]} 样本 × {X_sub.shape[1]} 维 ...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                init="pca", random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i in range(4):
        mask = y_sub == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=CLUSTER_COLORS[i], alpha=0.55, s=22,
                   edgecolors="none",
                   label=f"{CLUSTER_NAMES[i]}  (n={int(mask.sum())})")

    ax.set_xlabel("t-SNE 维度 1", fontsize=15)
    ax.set_ylabel("t-SNE 维度 2", fontsize=15)
    ax.set_title(f"基于 t-SNE 非线性流形的聚类结果分布图（k=4，采样 {sample_size}）",
                 fontsize=17, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=12, framealpha=0.93,
              markerscale=1.2)

    info = (f"perplexity = 30\n"
            f"迭代步数 = 1000\n"
            f"采样规模 = {sample_size} / {len(X_vals)}")
    ax.text(0.98, 0.02, info, transform=ax.transAxes,
            fontsize=12, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor="gray", alpha=0.93))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/3] 生成 1_特征工程_特征对比.png ...")
    plot_feature_comparison(out_dir / "1_特征工程_特征对比.png")

    print("\n[2/3] 生成 3_聚类结果_PCA降维.png ...")
    plot_pca_scatter(out_dir / "3_聚类结果_PCA降维.png")

    print("\n[3/3] 生成 3_聚类结果_tSNE降维.png ...")
    plot_tsne_scatter(out_dir / "3_聚类结果_tSNE降维.png")

    print(f"\n[OK] 全部图表已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
