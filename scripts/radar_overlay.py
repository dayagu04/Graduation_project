"""
第五章图 5-2：各簇核心特征对比雷达图
- 基于 cluster_centers_k4.csv 读取 4 个簇的质心
- 6 个关键特征：StudyHours, Extracurricular, EduTech, StressLevel,
                study_efficiency, academic_composite
- 4 个簇叠在同一张极坐标图上对比
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


KEY_FEATURES = [
    "StudyHours", "Extracurricular", "EduTech",
    "StressLevel", "study_efficiency", "academic_composite",
]

CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]  # 红、蓝、绿、紫

CLUSTER_NAMES = {
    0: "簇 0：男生课外活跃型",
    1: "簇 1：混合均衡型",
    2: "簇 2：女生学业专注型",
    3: "簇 3：女生课外活跃型",
}


def plot_radar_overlay(centers_path: Path, out_path: Path) -> None:
    centers_df = pd.read_csv(centers_path)
    if "cluster" in centers_df.columns:
        centers_df = centers_df.set_index("cluster")

    missing = [f for f in KEY_FEATURES if f not in centers_df.columns]
    if missing:
        raise RuntimeError(f"质心文件中缺失特征: {missing}")

    data = centers_df[KEY_FEATURES].values  # (4, 6)
    n_clusters, n_feats = data.shape

    # 角度：6 个特征均匀分布 + 闭合
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    for i in range(n_clusters):
        vals = data[i].tolist()
        vals += vals[:1]
        ax.plot(
            angles, vals,
            color=CLUSTER_COLORS[i], linewidth=2.2,
            marker="o", markersize=7,
            label=CLUSTER_NAMES.get(i, f"簇 {i}"),
        )
        ax.fill(angles, vals, color=CLUSTER_COLORS[i], alpha=0.15)

    # 角度刻度 = 特征名
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(KEY_FEATURES, fontsize=12)

    # 半径刻度 0..1
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="gray")
    ax.set_rlabel_position(90)  # 半径标签朝上

    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["polar"].set_alpha(0.35)

    ax.set_title("各簇核心特征对比雷达图（归一化后 0–1）",
                 fontsize=15, pad=26)

    # 图例放在图下方居中
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=2, fontsize=11, frameon=True, framealpha=0.95,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    centers_path = Path("output/cluster_centers_k4.csv")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    if not centers_path.exists():
        raise FileNotFoundError(
            f"找不到质心文件 {centers_path}，请先运行 scripts/generate_cluster_profiles.py"
        )

    out_path = out_dir / "4_群体画像_核心特征对比雷达图.png"
    plot_radar_overlay(centers_path, out_path)

    # 打印质心值便于论文引用
    df = pd.read_csv(centers_path).set_index("cluster")[KEY_FEATURES]
    print("\n=== 各簇 6 特征质心（归一化） ===")
    print(df.round(4).to_string())

    print(f"\n[OK] 雷达图已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
