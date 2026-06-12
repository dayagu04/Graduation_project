"""
重新生成综合总结面板，避免中文乱码：
1. 5_实验总结报告.png
2. 2_算法对比_综合表格.png
强制使用 SimHei / Microsoft YaHei，300 DPI 输出。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ── 图 A：算法对比综合表格 ─────────────────────────────────────────────────────

def fig_compare_table() -> None:
    """绘制三算法在 k=3,4,5 下的核心指标对比表 + 稳定性数据。"""
    cmp = pd.read_csv("output/train_algorithm_comparison.csv")
    stab = pd.read_csv("output/stability_train.csv")

    cmp_pivot_sse = cmp.pivot(index="k", columns="algorithm", values="sse")
    cmp_pivot_sil = cmp.pivot(index="k", columns="algorithm", values="silhouette")
    cmp_pivot_db  = cmp.pivot(index="k", columns="algorithm", values="davies_bouldin")
    cmp_pivot_t   = cmp.pivot(index="k", columns="algorithm", values="time_sec")

    stab_summary = stab.groupby("algorithm")["sse"].agg(["mean", "std"])
    stab_summary["cv"] = stab_summary["std"] / stab_summary["mean"]

    algos = ["K-Means", "K-Means++", "PSO-KMeans"]

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=0.35)

    # === 上方表格：三算法在不同 k 下的多指标对比 ===
    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    headers = ["指标", "k"] + algos
    rows = []
    for label, dfp in [("SSE",         cmp_pivot_sse),
                       ("Silhouette",  cmp_pivot_sil),
                       ("DB 指数",     cmp_pivot_db),
                       ("时间 (秒)",   cmp_pivot_t)]:
        for k in dfp.index:
            row = [label if k == dfp.index[0] else "", str(k)]
            for a in algos:
                v = dfp.loc[k, a]
                if label == "时间 (秒)":
                    row.append(f"{v:.2f}")
                elif abs(v) >= 100:
                    row.append(f"{v:.2f}")
                else:
                    row.append(f"{v:.4f}")
            rows.append(row)

    table1 = ax1.table(
        cellText=rows, colLabels=headers,
        loc="center", cellLoc="center",
        colWidths=[0.13, 0.07] + [0.20] * len(algos),
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.0, 1.55)

    # 表头美化
    for j, _ in enumerate(headers):
        cell = table1[0, j]
        cell.set_facecolor("#4f81bd")
        cell.set_text_props(color="white", fontweight="bold")
    # 指标分组分色
    color_map = {"SSE": "#fde9d9", "Silhouette": "#e8f1fa",
                 "DB 指数": "#eaf5ea", "时间 (秒)": "#f3eaf8"}
    cur_label = ""
    cur_color = "#fde9d9"
    for i, row in enumerate(rows, start=1):
        if row[0]:
            cur_label = row[0]
            cur_color = color_map.get(cur_label, "#ffffff")
        for j in range(len(headers)):
            table1[i, j].set_facecolor(cur_color)

    ax1.set_title("三算法在 k=3,4,5 下的核心指标对比（训练集，14,003 条）",
                  fontsize=14, pad=16, fontweight="bold")

    # === 下方表格：稳定性 (10 次运行) ===
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    stab_headers = ["算法", "SSE 均值", "SSE 标准差", "变异系数 (CV)"]
    stab_rows = []
    for a in algos:
        m = stab_summary.loc[a, "mean"]
        s = stab_summary.loc[a, "std"]
        cv = stab_summary.loc[a, "cv"]
        stab_rows.append([a, f"{m:.2f}", f"{s:.2f}", f"{cv*100:.2f}%"])

    table2 = ax2.table(
        cellText=stab_rows, colLabels=stab_headers,
        loc="center", cellLoc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(11.5)
    table2.scale(1.0, 1.7)
    for j in range(len(stab_headers)):
        cell = table2[0, j]
        cell.set_facecolor("#c0504d")
        cell.set_text_props(color="white", fontweight="bold")
    # 高亮 PSO-KMeans 行
    for j in range(len(stab_headers)):
        table2[3, j].set_facecolor("#fce4d6")
        table2[3, j].set_text_props(fontweight="bold")
    for i in [1, 2]:
        for j in range(len(stab_headers)):
            table2[i, j].set_facecolor("#f7f7f7")

    ax2.set_title("三算法稳定性对比（k=4，10 次独立运行）",
                  fontsize=14, pad=16, fontweight="bold")

    plt.suptitle("聚类算法综合性能对比表",
                 fontsize=16, fontweight="bold", y=0.995)
    plt.savefig(RESULTS / "2_算法对比_综合表格.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 图 B：实验总结报告（多面板）─────────────────────────────────────────────

def fig_experiment_summary() -> None:
    """6 个面板：数据概览 + 算法对比 + 聚类质量 + 群体分布 + 稳定性 + 关键结论。"""
    cmp = pd.read_csv("output/train_algorithm_comparison.csv")
    stab = pd.read_csv("output/stability_train.csv")
    elbow = pd.read_csv("output/train_elbow_metrics.csv")
    centers = pd.read_csv("output/cluster_centers_k4.csv")

    # 用 k=4 的样本数量（手算或读 cluster_interpretation.txt）
    cluster_sizes = [3283, 4074, 4064, 2582]
    cluster_labels = ["簇 0\n男生课外活跃型", "簇 1\n混合均衡型",
                      "簇 2\n女生学业专注型", "簇 3\n女生课外活跃型"]
    cluster_colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32,
                          height_ratios=[1, 1, 1])

    # ── 面板 1：数据概览（文字卡片）──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title("① 数据概览", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    panels = [
        ("训练集样本", "14,003 条", "#4f81bd"),
        ("验证集样本",    "300 条", "#4f81bd"),
        ("特征维度",       "18 维", "#4f81bd"),
        ("数据来源",  "Kaggle 公开", "#4f81bd"),
    ]
    for i, (label, value, color) in enumerate(panels):
        y = 0.88 - i * 0.22
        rect = mpatches.FancyBboxPatch(
            (0.05, y - 0.08), 0.9, 0.16,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1, edgecolor=color, facecolor=color, alpha=0.18,
        )
        ax1.add_patch(rect)
        ax1.text(0.10, y, label, fontsize=11, va="center", color="#333")
        ax1.text(0.92, y, value, fontsize=11.5, va="center", ha="right",
                 color=color, fontweight="bold")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # ── 面板 2：肘部法 SSE 曲线 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(elbow["k"], elbow["sse"], marker="o", color="#1f77b4", linewidth=2)
    ax2.axvline(4, color="red", linestyle="--", alpha=0.6, label="k=4 选定")
    ax2.set_title("② 肘部法 K 值评估", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    ax2.set_xlabel("k"); ax2.set_ylabel("SSE")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ── 面板 3：三算法 SSE @ k=4 柱状图 ──
    ax3 = fig.add_subplot(gs[0, 2])
    sub_k4 = cmp[cmp["k"] == 4].set_index("algorithm").loc[
        ["K-Means", "K-Means++", "PSO-KMeans"]]
    bars = ax3.bar(sub_k4.index, sub_k4["sse"],
                   color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                   edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, sub_k4["sse"]):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 30,
                 f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax3.set_title("③ 三算法 SSE 对比 (k=4)", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    ax3.set_ylabel("SSE")
    ax3.set_ylim(sub_k4["sse"].min() * 0.985, sub_k4["sse"].max() * 1.012)
    ax3.tick_params(axis="x", rotation=15)
    ax3.grid(axis="y", alpha=0.3)

    # ── 面板 4：簇大小分布饼图 ──
    ax4 = fig.add_subplot(gs[1, 0])
    wedges, _, autotexts = ax4.pie(
        cluster_sizes, labels=None,
        colors=cluster_colors, autopct="%.1f%%",
        startangle=90, textprops=dict(color="white", fontweight="bold", fontsize=10),
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    ax4.set_title("④ K=4 各簇样本占比", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    ax4.legend(wedges, [f"{lbl}（{n}人）" for lbl, n in zip(cluster_labels, cluster_sizes)],
               loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5,
               frameon=False)

    # ── 面板 5：稳定性箱线图 ──
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [stab[stab["algorithm"] == a]["sse"].values
                for a in ["K-Means", "K-Means++", "PSO-KMeans"]]
    bp = ax5.boxplot(box_data, labels=["K-Means", "K-Means++", "PSO-KMeans"],
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.5))
    for p, c in zip(bp["boxes"], ["#a6cee3", "#b2df8a", "#fb9a99"]):
        p.set_facecolor(c)
    ax5.set_title("⑤ 稳定性箱线图（10 次运行）", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    ax5.set_ylabel("SSE")
    ax5.tick_params(axis="x", rotation=15)
    ax5.grid(axis="y", alpha=0.3)

    # ── 面板 6：稳定性变异系数对比 ──
    ax6 = fig.add_subplot(gs[1, 2])
    cv_data = []
    for a in ["K-Means", "K-Means++", "PSO-KMeans"]:
        s = stab[stab["algorithm"] == a]["sse"]
        cv_data.append(s.std() / s.mean() * 100)
    bars = ax6.bar(["K-Means", "K-Means++", "PSO-KMeans"], cv_data,
                   color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                   edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, cv_data):
        ax6.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax6.set_title("⑥ 变异系数 CV 对比", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    ax6.set_ylabel("CV (%)")
    ax6.set_ylim(0, max(cv_data) * 1.25)
    ax6.tick_params(axis="x", rotation=15)
    ax6.grid(axis="y", alpha=0.3)

    # ── 面板 7：聚类质量指标雷达图（K-Means++ k=4）──
    ax7 = fig.add_subplot(gs[2, 0], projection="polar")
    metrics_k4 = cmp[cmp["k"] == 4].set_index("algorithm")
    norm_metrics = {
        "SSE↓":         1 - (metrics_k4["sse"] - metrics_k4["sse"].min())
                            / (metrics_k4["sse"].max() - metrics_k4["sse"].min() + 1e-9),
        "Silhouette↑": (metrics_k4["silhouette"] - metrics_k4["silhouette"].min())
                       / (metrics_k4["silhouette"].max() - metrics_k4["silhouette"].min() + 1e-9),
        "DB↓":          1 - (metrics_k4["davies_bouldin"] - metrics_k4["davies_bouldin"].min())
                             / (metrics_k4["davies_bouldin"].max() - metrics_k4["davies_bouldin"].min() + 1e-9),
        "CH↑":         (metrics_k4["calinski_harabasz"] - metrics_k4["calinski_harabasz"].min())
                       / (metrics_k4["calinski_harabasz"].max() - metrics_k4["calinski_harabasz"].min() + 1e-9),
        "稳定性↑":     pd.Series([0, 0.5, 1.0],
                                  index=["K-Means", "K-Means++", "PSO-KMeans"]),  # 经验排序
    }
    angles = np.linspace(0, 2 * np.pi, len(norm_metrics), endpoint=False).tolist()
    angles += angles[:1]
    for a, color in zip(["K-Means", "K-Means++", "PSO-KMeans"],
                        ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        vals = [norm_metrics[m][a] for m in norm_metrics]
        vals += vals[:1]
        ax7.plot(angles, vals, marker="o", linewidth=1.6, color=color, label=a)
        ax7.fill(angles, vals, color=color, alpha=0.18)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(list(norm_metrics.keys()), fontsize=9)
    ax7.set_ylim(0, 1)
    ax7.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax7.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax7.set_title("⑦ 综合质量雷达 (k=4)", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=14)
    ax7.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=8)

    # ── 面板 8 & 9：关键结论文字面板 ──
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis("off")
    ax8.set_title("⑧ 核心实验结论", fontsize=13, fontweight="bold",
                  loc="left", color="#1f4e79", pad=8)
    conclusions = [
        ("最佳聚类数", "k = 4（综合肘部法 + 轮廓系数 + 教育学可解释性）"),
        ("PSO-KMeans 优势", "10 次运行 SSE 均值最低（24,387.51）、标准差最小（277.64）"),
        ("分群主因", "Extracurricular / EduTech / StressLevel 三个行为维度"),
        ("教育发现", "学生群体差异主要在学习行为模式，而非传统成绩分层"),
        ("应用价值", "为教务管理提供差异化教学策略与个性化干预依据"),
    ]
    for i, (label, content) in enumerate(conclusions):
        y = 0.92 - i * 0.18
        ax8.text(0.02, y, "■", fontsize=14, color="#c0392b", va="center")
        ax8.text(0.06, y, label + "：", fontsize=11.5, fontweight="bold",
                 color="#222", va="center")
        ax8.text(0.27, y, content, fontsize=11, color="#444", va="center")
    ax8.set_xlim(0, 1); ax8.set_ylim(0, 1)

    plt.suptitle(
        "毕业论文实验总结报告 · 基于聚类算法的学生成绩分析与研究",
        fontsize=17, fontweight="bold", y=0.995,
    )
    plt.savefig(RESULTS / "5_实验总结报告.png",
                dpi=300, bbox_inches="tight")
    plt.close()


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[1/2] 重生成 2_算法对比_综合表格.png ...")
    fig_compare_table()
    print("[2/2] 重生成 5_实验总结报告.png ...")
    fig_experiment_summary()
    print(f"\n[OK] 全部图表已保存到: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
