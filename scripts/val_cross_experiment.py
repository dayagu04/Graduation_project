"""
跨数据集验证实验：在 val 集（300 条）上重复训练集的算法对比与稳定性测试
用于回应"模型泛化能力"和"小样本鲁棒性"两个论文关注点
产出：
  output/val_cross_comparison.csv      —— 单次截面性能（val 集）
  output/val_cross_stability.csv       —— 10 次独立运行 SSE
  output/val_cross_comparison.png      —— 4 指标雷达式对比
  output/val_cross_stability.png       —— 跨数据集稳定性箱线图
  output/val_cross_summary.csv         —— train vs val 算法 CV 综合表
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, VAL_CONFIG, prepare
from pso_kmeans import PSOKMeans


ALGOS = ["K-Means", "K-Means++", "PSO-KMeans"]
COLORS = {"K-Means": "#3b82f6", "K-Means++": "#10b981", "PSO-KMeans": "#ef4444"}


def _build_model(algo: str, k: int, seed: int, val_mode: bool):
    """val 集小样本 => 给 PSO 更大搜索预算；K-Means/K-Means++ 给公平的 n_init。"""
    if algo == "PSO-KMeans":
        if val_mode:
            return PSOKMeans(n_clusters=k, n_particles=50, pso_max_iter=100,
                             random_state=seed)
        return PSOKMeans(n_clusters=k, n_particles=20, pso_max_iter=30,
                         random_state=seed)
    if algo == "K-Means++":
        return KMeans(n_clusters=k, init="k-means++", n_init=20,
                      max_iter=300, random_state=seed)
    return KMeans(n_clusters=k, init="random", n_init=10,
                  max_iter=300, random_state=seed)


def single_pass(X, k: int, seed: int, val_mode: bool) -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        model = _build_model(algo, k, seed, val_mode)
        labels = model.fit_predict(X)
        rows.append({
            "algorithm": algo,
            "sse": float(model.inertia_),
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "n_samples": int(X.shape[0]),
        })
    return pd.DataFrame(rows)


def stability_pass(X, k: int, n_runs: int, seed_base: int, val_mode: bool) -> pd.DataFrame:
    rows = []
    for run in range(n_runs):
        seed = seed_base + run
        for algo in ALGOS:
            # 稳定性场景：把 K-Means / K-Means++ 的 n_init 设为 1
            if algo == "K-Means":
                model = KMeans(n_clusters=k, init="random", n_init=1,
                               max_iter=300, random_state=seed)
            elif algo == "K-Means++":
                model = KMeans(n_clusters=k, init="k-means++", n_init=1,
                               max_iter=300, random_state=seed)
            else:
                model = _build_model(algo, k, seed, val_mode)
            model.fit(X)
            rows.append({
                "algorithm": algo,
                "run": run + 1,
                "seed": seed,
                "sse": float(model.inertia_),
            })
    return pd.DataFrame(rows)


def plot_comparison(cmp_train: pd.DataFrame, cmp_val: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = [("sse",         "SSE (越小越好)",        True),
               ("silhouette",  "轮廓系数 (越大越好)",   False),
               ("davies_bouldin", "DB 指数 (越小越好)", True),
               ("calinski_harabasz", "CH 指数 (越大越好)", False)]

    for ax, (metric, label, _lower_better) in zip(axes.flat, metrics):
        x = np.arange(len(ALGOS))
        train_vals = [cmp_train[cmp_train.algorithm == a][metric].iloc[0] for a in ALGOS]
        val_vals = [cmp_val[cmp_val.algorithm == a][metric].iloc[0] for a in ALGOS]

        bw = 0.36
        ax.bar(x - bw/2, train_vals, bw, label="train (14003)",
               color=[COLORS[a] for a in ALGOS], alpha=0.85, edgecolor="white")
        ax.bar(x + bw/2, val_vals, bw, label="val (300)",
               color=[COLORS[a] for a in ALGOS], alpha=0.45,
               hatch="///", edgecolor="white")

        for i, v in enumerate(train_vals):
            ax.text(i - bw/2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
        for i, v in enumerate(val_vals):
            ax.text(i + bw/2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    plt.suptitle("三算法在 train / val 双数据集上的截面性能对比 (k=4)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stability(stab_train: pd.DataFrame, stab_val: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, title in [
        (axes[0], stab_train, "train 集 (n=14003) · 10 次独立运行 SSE 分布"),
        (axes[1], stab_val,   "val 集 (n=300) · 10 次独立运行 SSE 分布"),
    ]:
        data = [df[df.algorithm == a]["sse"].values for a in ALGOS]
        bp = ax.boxplot(data, labels=ALGOS, patch_artist=True, widths=0.55)
        for patch, algo in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.55)
        ax.set_title(title)
        ax.set_ylabel("SSE")
        ax.grid(axis="y", alpha=0.3)

        # 在每个箱上方标注 std
        for i, algo in enumerate(ALGOS, start=1):
            std = float(df[df.algorithm == algo]["sse"].std())
            mean = float(df[df.algorithm == algo]["sse"].mean())
            ax.text(i, ax.get_ylim()[1] * 0.98,
                    f"σ={std:.2f}\nμ={mean:.2f}",
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out = Path("output")
    out.mkdir(exist_ok=True)

    X_train, _, _ = prepare(TRAIN_CONFIG)
    X_val,   _, _ = prepare(VAL_CONFIG)
    X_train, X_val = X_train.values, X_val.values

    print(f"train: {X_train.shape}    val: {X_val.shape}")

    # ---- 单次截面 ----
    print("\n[1/2] 截面性能 (k=4, seed=42) ...")
    cmp_train = single_pass(X_train, k=4, seed=42, val_mode=False)
    cmp_train["dataset"] = "train"
    cmp_val = single_pass(X_val, k=4, seed=42, val_mode=True)
    cmp_val["dataset"] = "val"
    pd.concat([cmp_train, cmp_val]).to_csv(
        out / "val_cross_comparison.csv", index=False, encoding="utf-8-sig")
    plot_comparison(cmp_train, cmp_val, out / "val_cross_comparison.png")
    print("  train:")
    print(cmp_train.round(4).to_string(index=False))
    print("  val:")
    print(cmp_val.round(4).to_string(index=False))

    # ---- 10 次稳定性 ----
    print("\n[2/2] 稳定性 (k=4, n_runs=10) ...")
    stab_train = stability_pass(X_train, k=4, n_runs=10, seed_base=42, val_mode=False)
    stab_train["dataset"] = "train"
    stab_val = stability_pass(X_val, k=4, n_runs=10, seed_base=42, val_mode=True)
    stab_val["dataset"] = "val"
    pd.concat([stab_train, stab_val]).to_csv(
        out / "val_cross_stability.csv", index=False, encoding="utf-8-sig")
    plot_stability(stab_train, stab_val, out / "val_cross_stability.png")

    # ---- 综合 CV 表 ----
    summary_rows = []
    for ds_name, df in [("train", stab_train), ("val", stab_val)]:
        agg = df.groupby("algorithm")["sse"].agg(["mean", "std", "min", "max"])
        agg["cv"] = agg["std"] / agg["mean"]
        agg["dataset"] = ds_name
        summary_rows.append(agg.reset_index())
    summary = pd.concat(summary_rows)
    summary = summary[["dataset", "algorithm", "mean", "std", "cv", "min", "max"]]
    summary.to_csv(out / "val_cross_summary.csv", index=False, encoding="utf-8-sig")
    print("\n稳定性综合表:")
    print(summary.round(4).to_string(index=False))

    print(f"\n所有结果已保存到: {out}/")


if __name__ == "__main__":
    main()
