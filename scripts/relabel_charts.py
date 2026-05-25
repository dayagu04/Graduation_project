"""
统一为 6 张论文图表添加学术风格的子图字母标注 (a)(b)(c)(d)。
覆盖原文件，保持 300 DPI。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, VAL_CONFIG, prepare
from pso_kmeans import PSOKMeans

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ── 通用：在子图左上角加 (a)/(b)/... 学术标注 ────────────────────────────────

def add_subplot_label(ax, letter: str, *,
                      x: float = -0.10, y: float = 1.06,
                      fontsize: int = 16) -> None:
    """字母标注：Arial 加粗，与坐标轴刻度字号一致或略大。"""
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily="Arial",
        fontweight="bold",
        va="bottom", ha="left",
    )


# ── 图 1：KDE 核密度分布检验 (4 子图) ─────────────────────────────────────────

def fig_kde() -> None:
    df = pd.read_csv("data/train-data.csv")
    features = ["ExamScore", "StudyHours", "Attendance", "StressLevel"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    letters = ["a", "b", "c", "d"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna().astype(float)
        sk, kt = skew(data), kurtosis(data)

        sns.kdeplot(data=data, ax=ax, fill=True, color=colors[i],
                    linewidth=2.4, alpha=0.55)
        ax.axvline(data.mean(), color="black", linestyle="--",
                   linewidth=1.4, alpha=0.7, label=f"均值={data.mean():.2f}")

        ax.set_title(f"{feat} 核密度分布", fontsize=22, fontweight="bold")
        ax.set_xlabel(feat, fontsize=20)
        ax.set_ylabel("密度", fontsize=22)
        ax.tick_params(axis="both", labelsize=16)
        ax.text(0.97, 0.95,
                f"Skewness = {sk:+.3f}\nKurtosis = {kt:+.3f}\nN = {len(data)}",
                transform=ax.transAxes, fontsize=16,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.85))
        ax.legend(loc="upper left", fontsize=15)
        ax.grid(alpha=0.3)
        add_subplot_label(ax, letters[i], fontsize=22)

    plt.suptitle("训练集关键特征核密度估计（KDE）分布检验",
                 fontsize=26, y=1.00, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "1_特征工程_KDE分布检验.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 2：衍生特征分布 (4 子图) ─────────────────────────────────────────────────

def fig_derived() -> None:
    """(a) 成绩波动率  (b) 成绩趋势  (c) 总成绩  (d) 学习效率"""
    X_val, names_val, _ = prepare(VAL_CONFIG)
    X_tr,  names_tr,  _ = prepare(TRAIN_CONFIG)

    feat_data = [
        ("成绩波动率 (quiz_volatility)", X_val["quiz_volatility"], "#e74c3c"),
        ("成绩趋势 (quiz_trend)",        X_val["quiz_trend"],      "#3498db"),
        ("总成绩 (academic_composite)",  X_tr["academic_composite"], "#27ae60"),
        ("学习效率 (study_efficiency)",  X_tr["study_efficiency"], "#9b59b6"),
    ]
    letters = ["a", "b", "c", "d"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (title, vals, color) in enumerate(feat_data):
        ax = axes[i]
        vals = vals.dropna().astype(float)
        sk, kt = skew(vals), kurtosis(vals)

        sns.histplot(vals, kde=True, ax=ax, color=color,
                     alpha=0.55, edgecolor="white", linewidth=0.4,
                     stat="density", bins=30)
        ax.axvline(vals.mean(), color="black", linestyle="--",
                   linewidth=1.4, alpha=0.75, label=f"均值={vals.mean():.3f}")

        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_xlabel("归一化取值", fontsize=20)
        ax.set_ylabel("密度", fontsize=22)
        ax.tick_params(axis="both", labelsize=16)
        ax.text(0.97, 0.95,
                f"Skewness = {sk:+.3f}\nKurtosis = {kt:+.3f}\nN = {len(vals)}",
                transform=ax.transAxes, fontsize=16,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.85))
        ax.legend(loc="upper left", fontsize=15)
        ax.grid(alpha=0.3)
        add_subplot_label(ax, letters[i], fontsize=22)

    plt.suptitle("面向教育场景的衍生特征分布",
                 fontsize=26, y=1.00, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "1_特征工程_衍生特征分布.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 3：K 值评估（肘部法 + 轮廓系数）── 2 子图 ──────────────────────────────

def fig_k_eval() -> None:
    df = pd.read_csv("output/train_elbow_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(df["k"], df["sse"], marker="o", linewidth=2.8, color="#1f77b4",
                 markersize=11)
    axes[0].axvline(4, color="red", linestyle="--", linewidth=2.0, alpha=0.8,
                    label="k=4（选定）")
    axes[0].set_title("肘部法（Elbow Method, K-Means++）", fontsize=18, fontweight="bold", pad=12)
    axes[0].set_xlabel("k", fontsize=17, fontweight="bold")
    axes[0].set_ylabel("SSE", fontsize=17, fontweight="bold")
    axes[0].tick_params(axis="both", labelsize=15)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].yaxis.get_offset_text().set_fontsize(14)
    axes[0].legend(fontsize=15)
    axes[0].grid(alpha=0.3)
    add_subplot_label(axes[0], "a", fontsize=18)

    axes[1].plot(df["k"], df["silhouette"], marker="o", linewidth=2.8,
                 color="#ff7f0e", markersize=11)
    axes[1].axvline(4, color="red", linestyle="--", linewidth=2.0, alpha=0.8,
                    label="k=4（选定）")
    axes[1].set_title("轮廓系数曲线（Silhouette Score）", fontsize=18, fontweight="bold", pad=12)
    axes[1].set_xlabel("k", fontsize=17, fontweight="bold")
    axes[1].set_ylabel("Silhouette", fontsize=17, fontweight="bold")
    axes[1].tick_params(axis="both", labelsize=15)
    axes[1].legend(fontsize=15)
    axes[1].grid(alpha=0.3)
    add_subplot_label(axes[1], "b", fontsize=18)

    plt.suptitle("最佳 K 值评估（训练集）", fontsize=21, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "2_算法对比_K值评估.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    plt.close("all")


# ── 图 4：性能对比 (SSE / Silhouette / DB / CH) 2x2 子图 ───────────────────────

def fig_performance() -> None:
    df = pd.read_csv("output/train_algorithm_comparison.csv")
    algos = ["K-Means", "K-Means++", "PSO-KMeans"]
    colors = {"K-Means": "#1f77b4", "K-Means++": "#ff7f0e", "PSO-KMeans": "#2ca02c"}

    metrics = [
        ("sse",               "SSE",                "SSE（越小越好）",                 "a"),
        ("silhouette",        "Silhouette",         "轮廓系数（越大越好）",            "b"),
        ("davies_bouldin",    "Davies-Bouldin",     "DB 指数（越小越好）",             "c"),
        ("calinski_harabasz", "Calinski-Harabasz",  "CH 指数（越大越好）",             "d"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    axes = axes.flatten()

    for i, (col, ylabel, title, letter) in enumerate(metrics):
        ax = axes[i]
        for algo in algos:
            sub = df[df["algorithm"] == algo].sort_values("k")
            ax.plot(sub["k"], sub[col], marker="o", linewidth=2.8,
                    markersize=11, color=colors[algo], label=algo)
        ax.set_title(title, fontsize=24, fontweight="bold", pad=12)
        ax.set_xlabel("k", fontsize=22, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=22, fontweight="bold")
        ax.tick_params(axis="both", labelsize=18)
        if col in {"sse", "calinski_harabasz"}:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_fontsize(17)
        ax.legend(fontsize=18)
        ax.grid(alpha=0.3)
        add_subplot_label(ax, letter, fontsize=22)

    plt.suptitle("三算法性能对比（训练集，k=3,4,5）", fontsize=28, y=1.00,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "2_算法对比_性能指标.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 5：稳定性对比（箱线图 + 折线图） 2 子图 ────────────────────────────────

def fig_stability() -> None:
    df = pd.read_csv("output/stability_train.csv")
    algos = ["K-Means", "K-Means++", "PSO-KMeans"]
    box_colors = ["#a6cee3", "#b2df8a", "#fb9a99"]
    line_colors = {"K-Means": "#1f77b4", "K-Means++": "#2ca02c", "PSO-KMeans": "#d62728"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 16))

    # (a) 箱线图
    data_to_plot = [df[df["algorithm"] == a]["sse"].values for a in algos]
    bp = axes[0].boxplot(data_to_plot, labels=algos, patch_artist=True,
                         widths=0.55,
                         medianprops=dict(color="black", linewidth=2.0))
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#333")
    axes[0].set_ylabel("SSE", fontsize=22, fontweight="bold")
    axes[0].set_title("SSE 分布箱线图（50 次独立运行，k=4）", fontsize=22,
                      fontweight="bold", pad=12)
    axes[0].tick_params(axis="both", labelsize=20)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].yaxis.get_offset_text().set_fontsize(18)
    axes[0].grid(axis="y", alpha=0.3)
    add_subplot_label(axes[0], "a", fontsize=22)

    # (b) 折线图：每次运行的波动
    for algo in algos:
        sub = df[df["algorithm"] == algo].sort_values("run")
        axes[1].plot(sub["run"], sub["sse"], marker="o", linewidth=2.0,
                     markersize=7, alpha=0.9, color=line_colors[algo],
                     label=algo)
    axes[1].set_xlabel("运行编号 (Run)", fontsize=22, fontweight="bold")
    axes[1].set_ylabel("SSE", fontsize=22, fontweight="bold")
    axes[1].set_title("SSE 跨运行波动折线图", fontsize=22, fontweight="bold", pad=12)
    axes[1].set_xticks(np.arange(5, df["run"].max() + 1, 5))
    axes[1].tick_params(axis="both", labelsize=18)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].yaxis.get_offset_text().set_fontsize(18)
    axes[1].legend(fontsize=18)
    axes[1].grid(alpha=0.3)
    add_subplot_label(axes[1], "b", fontsize=22)

    plt.suptitle("三算法稳定性对比（训练集）", fontsize=28, y=1.00,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "2_算法对比_稳定性.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 6：均值偏离分析（4 个簇双向条形图） 2x2 子图 ──────────────────────────

KEY_FEATURES = [
    "StudyHours", "Extracurricular", "EduTech",
    "StressLevel", "study_efficiency", "academic_composite",
]

CLUSTER_NAMES = {
    0: "簇 0：男生课外活跃型",
    1: "簇 1：混合均衡型",
    2: "簇 2：女生学业专注型",
    3: "簇 3：女生课外活跃型",
}


def fig_deviation() -> None:
    # 读取论文第五章口径下的 PSO-KMeans (k=4) 聚类结果
    df = pd.read_csv("output/clustering_result_k4.csv")
    labels = df["cluster"].to_numpy()
    feat_df = df.drop(columns=["cluster"])
    feature_names = feat_df.columns.tolist()
    X_vals = feat_df.to_numpy()

    n_clusters = 4
    centers = np.vstack([X_vals[labels == i].mean(axis=0) for i in range(n_clusters)])
    cluster_sizes = [int((labels == i).sum()) for i in range(n_clusters)]

    idx = [feature_names.index(f) for f in KEY_FEATURES]
    center_vals = centers[:, idx]
    global_mean = X_vals[:, idx].mean(axis=0)
    deviations = center_vals - global_mean

    pos_color, neg_color = "#d62728", "#1f77b4"
    x_abs = np.abs(deviations).max()
    x_lim = x_abs * 1.32
    y_pos = np.arange(len(KEY_FEATURES))[::-1]
    letters = ["a", "b", "c", "d"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    for i in range(n_clusters):
        ax = axes[i]
        dev = deviations[i]
        colors = [pos_color if v >= 0 else neg_color for v in dev]
        bars = ax.barh(y_pos, dev, color=colors, edgecolor="black",
                       linewidth=0.8, alpha=0.88)
        ax.axvline(0, color="black", linewidth=1.2)

        for bar, val in zip(bars, dev):
            offset = x_abs * 0.018
            ha = "left" if val >= 0 else "right"
            ax.text(bar.get_width() + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=17,
                    fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(KEY_FEATURES, fontsize=20, fontweight="bold")
        ax.tick_params(axis="x", labelsize=16)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_xlabel("相对全局均值的偏离量（归一化空间）", fontsize=18, fontweight="bold")
        ax.set_title(
            f"{CLUSTER_NAMES[i]}  (n={cluster_sizes[i]}, "
            f"{cluster_sizes[i]/len(X_vals)*100:.1f}%)",
            fontsize=20, pad=10, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.text(-x_lim * 0.97, -0.55, "低于均值", fontsize=16, color=neg_color,
                ha="left", va="center", fontweight="bold")
        ax.text(x_lim * 0.97, -0.55, "高于均值", fontsize=16, color=pos_color,
                ha="right", va="center", fontweight="bold")
        add_subplot_label(ax, letters[i], fontsize=22)

    plt.suptitle("PSO-KMeans (k=4) 各簇关键特征质心相对全局均值偏离分析",
                 fontsize=26, y=1.00, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "4_群体画像_均值偏离分析图.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[1/6] 重生成 1_特征工程_KDE分布检验.png ...")
    fig_kde()
    print("[2/6] 重生成 1_特征工程_衍生特征分布.png ...")
    fig_derived()
    print("[3/6] 重生成 2_算法对比_K值评估.png ...")
    fig_k_eval()
    print("[4/6] 重生成 2_算法对比_性能指标.png ...")
    fig_performance()
    print("[5/6] 重生成 2_算法对比_稳定性.png ...")
    fig_stability()
    print("[6/6] 重生成 4_群体画像_均值偏离分析图.png ...")
    fig_deviation()
    print(f"\n[OK] 全部图表已保存到: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
