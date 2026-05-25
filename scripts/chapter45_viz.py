"""
第四/五章补充配图
1. 4_群体画像_均值偏离分析图.png —— 各簇 6 个关键特征相对全局均值的偏离度
2. 2_算法对比_PSO收敛曲线.png   —— PSO 全局最优 SSE 随迭代次数的收敛曲线
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, prepare
from pso_kmeans import PSOKMeans

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


KEY_FEATURES = [
    "StudyHours", "Extracurricular", "EduTech",
    "StressLevel", "study_efficiency", "academic_composite",
]


# ── 图 1：各簇均值偏离分析图（双向条形图）────────────────────────────────────

def plot_deviation_bars(out_path: Path) -> None:
    X, feature_names, _ = prepare(TRAIN_CONFIG)
    X_vals = X.values

    # k=4 PSO-KMeans 融合聚类（与论文第五章口径保持一致）
    model = PSOKMeans(
        n_clusters=4, n_particles=20, pso_max_iter=30,
        kmeans_max_iter=300, random_state=42,
    )
    model.fit(X_vals)
    labels = model.labels_

    # 按簇取样本均值（与 cluster_centers_k4.csv 构建方式一致）
    n_clusters = 4
    centers = np.vstack([X_vals[labels == i].mean(axis=0) for i in range(n_clusters)])
    cluster_sizes = [int((labels == i).sum()) for i in range(n_clusters)]

    # 只挑 6 个关键特征
    missing = [f for f in KEY_FEATURES if f not in feature_names]
    if missing:
        raise RuntimeError(f"特征缺失: {missing}")
    idx = [feature_names.index(f) for f in KEY_FEATURES]
    feat_labels = KEY_FEATURES

    center_vals = centers[:, idx]            # (4, 6)
    global_mean = X_vals[:, idx].mean(axis=0)  # (6,)
    deviations = center_vals - global_mean   # (4, 6) 偏离量

    # 控制台打印
    print("\n=== 全局均值 (归一化后) ===")
    for f, m in zip(feat_labels, global_mean):
        print(f"  {f:<22s} {m:.4f}")
    print("\n=== 各簇偏离量 ===")
    dev_df = pd.DataFrame(deviations, columns=feat_labels,
                          index=[f"簇 {i}" for i in range(n_clusters)])
    print(dev_df.round(4).to_string())

    # 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()

    pos_color = "#d62728"   # 正偏（高于均值）→ 暖色
    neg_color = "#1f77b4"   # 负偏（低于均值）→ 冷色

    # 所有子图共享 x 轴范围，便于横向对比
    x_abs = np.abs(deviations).max()
    x_lim = x_abs * 1.25

    # y 轴位置，倒序让第一特征在最上方
    y_pos = np.arange(len(feat_labels))[::-1]

    cluster_names = {
        0: "簇 0：男生课外活跃型",
        1: "簇 1：混合均衡型",
        2: "簇 2：女生学业专注型",
        3: "簇 3：女生课外活跃型",
    }

    for i in range(n_clusters):
        ax = axes[i]
        dev = deviations[i]
        colors = [pos_color if v >= 0 else neg_color for v in dev]

        bars = ax.barh(y_pos, dev, color=colors, edgecolor="black",
                       linewidth=0.6, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.0)

        # 在条末端标数值
        for bar, val in zip(bars, dev):
            offset = x_abs * 0.015
            x = bar.get_width()
            ha = "left" if val >= 0 else "right"
            ax.text(x + (offset if val >= 0 else -offset),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=13)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=14)
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_xlabel("相对全局均值的偏离量（归一化空间）", fontsize=14)
        ax.set_title(
            f"{cluster_names[i]}  (n={cluster_sizes[i]}, {cluster_sizes[i]/len(X_vals)*100:.1f}%)",
            fontsize=15, pad=10, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # 左右两侧加方向提示
        ax.text(-x_lim * 0.97, -0.55, "低于均值",
                fontsize=12, color=neg_color, ha="left", va="center",
                fontweight="bold")
        ax.text(x_lim * 0.97, -0.55, "高于均值",
                fontsize=12, color=pos_color, ha="right", va="center",
                fontweight="bold")

    plt.suptitle(
        "PSO-KMeans (k=4) 各簇关键特征质心相对全局均值偏离分析",
        fontsize=19, y=1.00, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── 图 2：PSO 收敛曲线图 ──────────────────────────────────────────────────────

def plot_pso_convergence(out_path: Path) -> None:
    X, _, _ = prepare(TRAIN_CONFIG)
    X_vals = X.values

    model = PSOKMeans(
        n_clusters=4, n_particles=20, pso_max_iter=30,
        kmeans_max_iter=100, random_state=42,
    )
    model.fit(X_vals)
    history = model.convergence_history_  # 长度 = pso_max_iter + 1
    final_sse = model.inertia_            # K-Means 精细化后的最终 SSE

    iters = np.arange(len(history))  # 0..30

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(iters, history, marker="o", markersize=7,
            linewidth=2.6, color="#1f4e79",
            label="PSO 全局最优 SSE")
    ax.fill_between(iters, history, history.max(),
                    alpha=0.12, color="#1f4e79")

    # 初始点 & 终点标注
    ax.scatter([iters[0]], [history[0]], color="#d62728",
               s=130, zorder=5, label=f"初始化 SSE = {history[0]:.2f}")
    ax.scatter([iters[-1]], [history[-1]], color="#2ca02c",
               s=130, zorder=5, label=f"PSO 收敛 SSE = {history[-1]:.2f}")
    ax.axhline(final_sse, color="gray", linestyle="--", linewidth=1.6,
               label=f"K-Means 精细化后 SSE = {final_sse:.2f}")

    improve = history[0] - history[-1]
    improve_pct = improve / history[0] * 100 if history[0] else 0.0
    textstr = (
        f"粒子数 N = 20\n"
        f"最大迭代 T = 30\n"
        f"首代 SSE = {history[0]:.2f}\n"
        f"末代 SSE = {history[-1]:.2f}\n"
        f"下降量   = {improve:.2f}\n"
        f"下降比例 = {improve_pct:.2f}%"
    )
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes,
            fontsize=13, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor="gray", alpha=0.92))

    ax.set_xlabel("PSO 迭代次数", fontsize=15)
    ax.set_ylabel("全局最优适应度 (SSE)", fontsize=15)
    ax.set_title("PSO-KMeans 全局最优 SSE 收敛曲线（k=4，训练集）",
                 fontsize=17, pad=14, fontweight="bold")
    ax.set_xticks(np.arange(0, len(history), 2))
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.35, linestyle="--")
    ax.legend(loc="center right", fontsize=12.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 同时把收敛数据存一份 CSV，方便论文表格引用
    pd.DataFrame({"iteration": iters, "global_best_sse": history}) \
        .to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/2] 生成 4_群体画像_均值偏离分析图.png ...")
    plot_deviation_bars(out_dir / "4_群体画像_均值偏离分析图.png")

    print("\n[2/2] 生成 2_算法对比_PSO收敛曲线.png ...")
    plot_pso_convergence(out_dir / "2_算法对比_PSO收敛曲线.png")

    print(f"\n[OK] 全部图表已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
