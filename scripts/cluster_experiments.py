"""
聚类实验流水线（通用版）
- 不依赖任何固定列名，通过 feature_engineering.prepare() 获取特征
- 支持双数据集模式：val 集验证改进效果，train 集输出最终结论
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, VAL_CONFIG, DatasetConfig, prepare
from pso_kmeans import PSOKMeans


# ── 评估单个 (算法, k) 组合 ────────────────────────────────────────────────────

def _run_one(X, algo_name: str, k: int, random_state: int, pso_kwargs: dict = None) -> dict:
    start = time.perf_counter()
    if algo_name == "PSO-KMeans":
        kw = pso_kwargs or {}
        model = PSOKMeans(n_clusters=k, random_state=random_state, **kw)
    elif algo_name == "K-Means++":
        model = KMeans(n_clusters=k, init="k-means++", n_init=20,
                       max_iter=300, random_state=random_state)
    else:  # K-Means
        model = KMeans(n_clusters=k, init="random", n_init=10,
                       max_iter=300, random_state=random_state)

    labels = model.fit_predict(X)
    elapsed = time.perf_counter() - start

    return {
        "algorithm": algo_name,
        "k": k,
        "sse": float(model.inertia_),
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "time_sec": elapsed,
    }


# ── k 范围评估（肘部法 + 轮廓系数）─────────────────────────────────────────────

def evaluate_k_range(X, k_min: int, k_max: int, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        rows.append(_run_one(X, "K-Means++", k, random_state))
    return pd.DataFrame(rows)


# ── 三算法对比 ─────────────────────────────────────────────────────────────────

def compare_algorithms(X, k_values: List[int], random_state: int = 42, pso_kwargs: dict = None) -> pd.DataFrame:
    rows = []
    for k in k_values:
        for algo in ["K-Means", "K-Means++", "PSO-KMeans"]:
            rows.append(_run_one(X, algo, k, random_state, pso_kwargs))
    return pd.DataFrame(rows).sort_values(["k", "algorithm"]).reset_index(drop=True)


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_k_range(metrics_df: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(metrics_df["k"], metrics_df["sse"], marker="o")
    axes[0].set_title("Elbow Method (K-Means++)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("SSE"); axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="orange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}elbow_silhouette.png", dpi=150)
    plt.close()


def plot_comparison(compare_df: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    algos = compare_df["algorithm"].unique()
    k_vals = sorted(compare_df["k"].unique())
    metrics = ["sse", "silhouette", "davies_bouldin", "calinski_harabasz"]
    labels  = ["SSE", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, label in zip(axes.flat, metrics, labels):
        for algo in algos:
            sub = compare_df[compare_df["algorithm"] == algo]
            ax.plot(sub["k"], sub[metric], marker="o", label=algo)
        ax.set_title(label); ax.set_xlabel("k"); ax.set_ylabel(label)
        ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("Algorithm Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}algorithm_comparison.png", dpi=150)
    plt.close()


# ── 主入口 ─────────────────────────────────────────────────────────────────────

def run_experiment(
    cfg: DatasetConfig,
    out_dir: Path,
    k_min: int,
    k_max: int,
    compare_k: List[int],
    random_state: int,
    prefix: str,
    pso_kwargs: dict = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"数据集: {cfg.path}  前缀: {prefix or '(无)'}")
    if pso_kwargs:
        print(f"PSO 参数: {pso_kwargs}")
    print(f"{'='*60}")

    X, names, _ = prepare(cfg)
    print(f"样本: {len(X)}  特征 ({len(names)}): {names}")

    # 1. k 范围评估
    print(f"\n[1/2] 评估 k={k_min}..{k_max} ...")
    k_df = evaluate_k_range(X.values, k_min, k_max, random_state)
    k_df.to_csv(out_dir / f"{prefix}elbow_metrics.csv", index=False, encoding="utf-8-sig")
    plot_k_range(k_df, out_dir, prefix)
    print(k_df[["k", "sse", "silhouette"]].to_string(index=False))

    # 2. 三算法对比
    print(f"\n[2/2] 三算法对比 k={compare_k} ...")
    cmp_df = compare_algorithms(X.values, compare_k, random_state, pso_kwargs)
    cmp_df.to_csv(out_dir / f"{prefix}algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    plot_comparison(cmp_df, out_dir, prefix)
    print(cmp_df.to_string(index=False))

    print(f"\n输出已保存到: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="聚类实验工具（双数据集）")
    parser.add_argument("--mode",   choices=["train", "val", "both"], default="both",
                        help="运行模式：train/val/both")
    parser.add_argument("--out",    default="output", help="输出目录")
    parser.add_argument("--k-min",  type=int, default=2)
    parser.add_argument("--k-max",  type=int, default=8)
    parser.add_argument("--compare-k", default="3,4,5", help="对比实验的 k 值，逗号分隔")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_k = [int(x.strip()) for x in args.compare_k.split(",")]

    # val 集用强 PSO 参数（小数据集，搜索充分）
    val_pso = {"n_particles": 50, "pso_max_iter": 100}

    if args.mode in ("val", "both"):
        run_experiment(VAL_CONFIG,   out_dir, args.k_min, args.k_max,
                       compare_k, args.seed, prefix="val_", pso_kwargs=val_pso)

    if args.mode in ("train", "both"):
        run_experiment(TRAIN_CONFIG, out_dir, args.k_min, args.k_max,
                       compare_k, args.seed, prefix="train_")


if __name__ == "__main__":
    main()
