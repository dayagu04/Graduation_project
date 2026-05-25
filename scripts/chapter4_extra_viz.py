"""
第四章补充配图
1. 2_算法对比_K4轮廓系数分布图.png  —— K-Means++ k=4 轮廓系数刀刃图
2. 2_算法对比_PSOKMeans算法流程图.png —— PSO-KMeans 执行流程图（matplotlib 手绘）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, prepare

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]


# ── 图 1：轮廓系数刀刃图 ───────────────────────────────────────────────────────

def plot_silhouette(out_path: Path) -> None:
    X, _, _ = prepare(TRAIN_CONFIG)
    X_vals = X.values

    kmeans = KMeans(
        n_clusters=4, init="k-means++", n_init=20,
        max_iter=300, random_state=42,
    )
    labels = kmeans.fit_predict(X_vals)

    # 14003 样本两两距离太大，采样加速
    sample_size = min(5000, len(X_vals))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_vals), size=sample_size, replace=False)
    X_sub, y_sub = X_vals[idx], labels[idx]

    sil_mean = silhouette_score(X_sub, y_sub)
    sil_vals = silhouette_samples(X_sub, y_sub)

    n_clusters = 4
    fig, ax = plt.subplots(figsize=(12, 8))

    y_lower = 10
    tick_positions = []
    for i in range(n_clusters):
        cluster_sil = np.sort(sil_vals[y_sub == i])
        size_i = len(cluster_sil)
        y_upper = y_lower + size_i

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_sil,
            facecolor=CLUSTER_COLORS[i], edgecolor=CLUSTER_COLORS[i],
            alpha=0.78,
        )
        # 簇内均值 + 占比（放在左侧，避免与刀刃重叠）
        cluster_mean = cluster_sil.mean()
        ax.text(
            -0.12, y_lower + size_i / 2,
            f"Cluster {i}\n(n={size_i}, μ={cluster_mean:.3f})",
            ha="right", va="center", fontsize=14,
            color=CLUSTER_COLORS[i], fontweight="bold",
        )
        tick_positions.append(y_lower + size_i / 2)
        y_lower = y_upper + 10

    ax.axvline(sil_mean, color="red", linestyle="--", linewidth=2.2,
               label=f"平均轮廓系数 = {sil_mean:.4f}")
    ax.axvline(0, color="black", linewidth=1.2, alpha=0.5)

    ax.set_xlabel("轮廓系数 (Silhouette Coefficient)", fontsize=17)
    ax.set_ylabel("样本（按簇分组并升序排列）", fontsize=17)
    ax.set_title(
        f"K-Means++ (k=4) 各簇轮廓系数分布（采样 {sample_size} / {len(X_vals)}）",
        fontsize=19, pad=16, fontweight="bold",
    )

    # 扩大左侧空间，确保文字不越界
    x_min = min(-0.25, sil_vals.min() * 1.1)
    x_max = max(0.6, sil_vals.max() * 1.05)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_lower)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=15, framealpha=0.95)

    # 额外注释：正/负轮廓系数含义（移到右上角，避免遮挡刀刃）
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  平均轮廓系数: {sil_mean:.4f}")
    for i in range(n_clusters):
        cs = sil_vals[y_sub == i]
        neg_pct = (cs < 0).mean() * 100
        print(f"  簇 {i}: n={len(cs):>5d}  均值={cs.mean():+.4f}  "
              f"负样本占比={neg_pct:.2f}%")


# ── 图 2：PSO-KMeans 算法流程图 ────────────────────────────────────────────────

def plot_flowchart(out_path: Path) -> None:
    # 节点定义：(标签, 形状, 颜色) —— 形状：ellipse/rect/diamond
    nodes = [
        ("开始",                        "ellipse", "#9bbb59"),    # 0
        ("初始化粒子群\n(位置 P、速度 V)",  "rect",    "#4f81bd"),    # 1
        ("计算各粒子 SSE 适应度\nf(P_i)",   "rect",    "#4f81bd"),    # 2
        ("更新 pbest 与 gbest",          "rect",    "#4f81bd"),    # 3
        ("更新粒子速度 V\n更新粒子位置 P",  "rect",    "#4f81bd"),    # 4
        ("是否达到最大\n迭代次数 T ?",      "diamond", "#f79646"),    # 5
        ("输出全局最优解\ngbest",          "rect",    "#8064a2"),    # 6
        ("以 gbest 作为\nK-Means 初始质心", "rect",   "#8064a2"),    # 7
        ("K-Means 局部迭代\n精细化调整中心", "rect",   "#8064a2"),    # 8
        ("输出最终聚类结果\n(labels, centers)", "rect", "#c0504d"),   # 9
        ("结束",                        "ellipse", "#9bbb59"),    # 10
    ]

    # 位置 (x, y)，画布大小 10 × 15，y 从上往下
    pos = {
        0:  (5, 14.3),
        1:  (5, 12.8),
        2:  (5, 11.2),
        3:  (5,  9.6),
        4:  (5,  8.0),
        5:  (5,  6.2),
        6:  (5,  4.4),
        7:  (5,  3.0),
        8:  (5,  1.6),
        9:  (5,  0.1),
        10: (5, -1.4),
    }

    node_w_rect, node_h_rect = 3.4, 1.0
    node_w_ell,  node_h_ell  = 2.2, 0.8
    node_w_dia,  node_h_dia  = 3.6, 1.3

    fig, ax = plt.subplots(figsize=(9, 14.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.5, 15.5)
    ax.axis("off")

    # 绘制节点
    for i, (label, shape, color) in enumerate(nodes):
        x, y = pos[i]
        if shape == "rect":
            box = FancyBboxPatch(
                (x - node_w_rect / 2, y - node_h_rect / 2),
                node_w_rect, node_h_rect,
                boxstyle="round,pad=0.06,rounding_size=0.15",
                linewidth=1.4, edgecolor="#333333",
                facecolor=color, alpha=0.9,
            )
            ax.add_patch(box)
        elif shape == "ellipse":
            ell = mpatches.Ellipse(
                (x, y), node_w_ell, node_h_ell,
                linewidth=1.4, edgecolor="#333333",
                facecolor=color, alpha=0.9,
            )
            ax.add_patch(ell)
        elif shape == "diamond":
            diamond = mpatches.Polygon(
                [
                    (x, y + node_h_dia / 2),
                    (x + node_w_dia / 2, y),
                    (x, y - node_h_dia / 2),
                    (x - node_w_dia / 2, y),
                ],
                closed=True,
                linewidth=1.4, edgecolor="#333333",
                facecolor=color, alpha=0.9,
            )
            ax.add_patch(diamond)

        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

    # 绘制带箭头的连线
    def add_arrow(src, dst, label=None, curved=False, offset=0):
        x1, y1 = pos[src]
        x2, y2 = pos[dst]

        # 根据形状调整起止点
        def edge_point(node_idx, x, y, toward_x, toward_y):
            shape = nodes[node_idx][1]
            if shape == "rect":
                h = node_h_rect / 2
            elif shape == "ellipse":
                h = node_h_ell / 2
            else:
                h = node_h_dia / 2
            if toward_y > y:
                return x, y + h
            elif toward_y < y:
                return x, y - h
            return x, y

        if not curved:
            sx, sy = edge_point(src, x1, y1, x2, y2)
            ex, ey = edge_point(dst, x2, y2, x1, y1)
            arrow = FancyArrowPatch(
                (sx, sy), (ex, ey),
                arrowstyle="-|>", mutation_scale=16,
                linewidth=1.5, color="#333333",
            )
            ax.add_patch(arrow)
            if label:
                ax.text((sx + ex) / 2 + 0.15, (sy + ey) / 2,
                        label, fontsize=10, color="#c0392b", va="center")
        else:
            # 回环：判定框(5) 右侧向上回到步骤 2
            sx, sy = x1 + node_w_dia / 2, y1
            ex, ey = x2 + node_w_rect / 2, y2
            connection = f"arc3,rad={0.45 if offset>=0 else -0.45}"
            arrow = FancyArrowPatch(
                (sx, sy), (ex, ey),
                arrowstyle="-|>", mutation_scale=16,
                linewidth=1.5, color="#c0392b",
                connectionstyle=connection,
            )
            ax.add_patch(arrow)
            if label:
                ax.text(sx + 1.4, (sy + ey) / 2, label,
                        fontsize=10.5, color="#c0392b",
                        ha="center", va="center", fontweight="bold")

    # 主干连接
    add_arrow(0, 1)
    add_arrow(1, 2)
    add_arrow(2, 3)
    add_arrow(3, 4)
    add_arrow(4, 5)
    add_arrow(5, 6, label="是")
    add_arrow(6, 7)
    add_arrow(7, 8)
    add_arrow(8, 9)
    add_arrow(9, 10)

    # 判定框“否” → 回到步骤 2（曲线回环）
    add_arrow(5, 2, label="否，迭代 t ← t+1", curved=True, offset=1)

    # 阶段分组标注
    ax.text(0.3, 11.2, "PSO 阶段", fontsize=12, color="#1f4e79",
            fontweight="bold", rotation=90, va="center")
    ax.text(0.3, 3.0, "K-Means\n精细化", fontsize=12, color="#4f2d78",
            fontweight="bold", rotation=90, va="center", ha="center")

    ax.set_title("PSO-KMeans 融合聚类算法执行流程图",
                 fontsize=15, pad=15, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    print("[1/2] 生成 2_算法对比_K4轮廓系数分布图.png ...")
    plot_silhouette(out_dir / "2_算法对比_K4轮廓系数分布图.png")

    print("\n[2/2] 生成 2_算法对比_PSOKMeans算法流程图.png ...")
    plot_flowchart(out_dir / "2_算法对比_PSOKMeans算法流程图.png")

    print(f"\n[OK] 全部图表已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
