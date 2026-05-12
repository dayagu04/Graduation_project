"""
特征工程可视化脚本（第三章配图）
1. KDE 核密度分布（偏度/峰度标注）
2. 半三角 Pearson 相关性矩阵
3. Min-Max 归一化后的全局箱线图
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 图 1：KDE 核密度分布 ───────────────────────────────────────────────────────

def plot_kde(df: pd.DataFrame, out_path: Path) -> None:
    features = ["ExamScore", "StudyHours", "Attendance", "StressLevel"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna().astype(float)
        sk = skew(data)
        kt = kurtosis(data)  # 默认 Fisher 超额峰度（正态=0）

        sns.kdeplot(
            data=data, ax=ax, fill=True,
            color=colors[i], linewidth=2, alpha=0.55,
        )
        ax.axvline(data.mean(), color="black", linestyle="--",
                   linewidth=1, alpha=0.6, label=f"均值={data.mean():.2f}")

        ax.set_title(f"{feat} 核密度分布", fontsize=13)
        ax.set_xlabel(feat, fontsize=11)
        ax.set_ylabel("密度", fontsize=11)

        textstr = (
            f"Skewness = {sk:+.3f}\n"
            f"Kurtosis = {kt:+.3f}\n"
            f"N = {len(data)}"
        )
        ax.text(
            0.97, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10.5,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("训练集关键特征核密度估计（KDE）分布检验",
                 fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 2：半三角 Pearson 相关性矩阵 ───────────────────────────────────────────

def plot_corr_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method="pearson")

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.6, linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson 相关系数"},
        annot_kws={"size": 9},
        ax=ax,
    )
    ax.set_title("训练集特征 Pearson 相关性矩阵（下三角）",
                 fontsize=14, pad=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 3：Min-Max 归一化后的全局箱线图 ────────────────────────────────────────

def plot_global_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    num_df = df.select_dtypes(include="number").copy()
    # 剔除全常数列，避免归一化后全 0 / NaN
    num_df = num_df.loc[:, num_df.std() > 1e-9]

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(num_df),
        columns=num_df.columns,
    )

    # 统计每列离群点数量（IQR 规则）
    outlier_counts = {}
    for col in scaled.columns:
        q1, q3 = scaled[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[col] = int(((scaled[col] < lo) | (scaled[col] > hi)).sum())

    fig, ax = plt.subplots(figsize=(15, 7))
    bp = ax.boxplot(
        [scaled[c].values for c in scaled.columns],
        labels=list(scaled.columns),
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o", markerfacecolor="#d62728",
            markersize=3.5, markeredgecolor="#8b0000",
            alpha=0.5,
        ),
        medianprops=dict(color="black", linewidth=1.6),
        boxprops=dict(facecolor="#a6cee3", edgecolor="#1f4e79", linewidth=1.1),
        whiskerprops=dict(color="#1f4e79", linewidth=1.1),
        capprops=dict(color="#1f4e79", linewidth=1.1),
    )

    # 在每个箱子上方标注离群点数量
    for i, col in enumerate(scaled.columns, start=1):
        n = outlier_counts[col]
        if n > 0:
            ax.text(i, 1.03, f"n={n}", ha="center", va="bottom",
                    fontsize=8, color="#8b0000", rotation=0)

    ax.set_title("训练集特征全局箱线图（Min-Max 归一化后，红点为离群点）",
                 fontsize=14, pad=18)
    ax.set_xlabel("特征", fontsize=12)
    ax.set_ylabel("归一化取值 (0–1)", fontsize=12)
    ax.set_ylim(-0.08, 1.12)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    # 图例说明
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#a6cee3", edgecolor="#1f4e79", label="IQR 箱体"),
        Line2D([0], [0], color="black", linewidth=1.6, label="中位数"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markeredgecolor="#8b0000", markersize=6, label="离群点 (>1.5·IQR)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 控制台打印离群点统计
    print("\n离群点统计（IQR 规则，归一化后）:")
    for col, n in sorted(outlier_counts.items(), key=lambda x: -x[1]):
        print(f"  {col:<24s}  {n:>5d}  ({n/len(scaled)*100:.2f}%)")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    data_path = Path("data/train-data.csv")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path.resolve()}")

    df = pd.read_csv(data_path)
    print(f"数据集: {len(df)} 行 × {len(df.columns)} 列")
    print(f"数值列: {df.select_dtypes(include='number').shape[1]} 个")

    print("\n[1/3] 生成 KDE 核密度分布图...")
    plot_kde(df, out_dir / "1_特征工程_KDE分布检验.png")

    print("[2/3] 生成半三角 Pearson 相关性矩阵...")
    plot_corr_heatmap(df, out_dir / "1_特征工程_半三角相关性矩阵.png")

    print("[3/3] 生成全局箱线图...")
    plot_global_boxplot(df, out_dir / "1_特征工程_全局箱线图.png")

    print(f"\n[OK] 全部图表已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
