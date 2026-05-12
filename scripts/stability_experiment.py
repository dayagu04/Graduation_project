"""
PSO-KMeans 稳定性实验
对比 K-Means（随机初始化）vs PSO-KMeans 在多次运行下的 SSE 方差
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, VAL_CONFIG, prepare
from pso_kmeans import PSOKMeans


def stability_experiment(X, k: int, n_runs: int = 10, random_seed_base: int = 42):
    """
    多次运行实验，对比 K-Means 和 PSO-KMeans 的稳定性

    Returns
    -------
    results : pd.DataFrame
        每次运行的 SSE 记录
    """
    results = []

    for run in range(n_runs):
        seed = random_seed_base + run

        # K-Means (random init)
        km = KMeans(n_clusters=k, init='random', n_init=1, max_iter=300, random_state=seed)
        km.fit(X)
        results.append({
            'algorithm': 'K-Means',
            'run': run + 1,
            'seed': seed,
            'sse': km.inertia_
        })

        # K-Means++ (deterministic init)
        kmp = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, random_state=seed)
        kmp.fit(X)
        results.append({
            'algorithm': 'K-Means++',
            'run': run + 1,
            'seed': seed,
            'sse': kmp.inertia_
        })

        # PSO-KMeans
        pso = PSOKMeans(n_clusters=k, n_particles=20, pso_max_iter=30, random_state=seed)
        pso.fit(X)
        results.append({
            'algorithm': 'PSO-KMeans',
            'run': run + 1,
            'seed': seed,
            'sse': pso.inertia_
        })

    return pd.DataFrame(results)


def plot_stability(df: pd.DataFrame, out_path: Path, dataset_name: str):
    """绘制稳定性对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：箱线图
    algos = ['K-Means', 'K-Means++', 'PSO-KMeans']
    data_to_plot = [df[df['algorithm'] == a]['sse'].values for a in algos]

    bp = axes[0].boxplot(data_to_plot, labels=algos, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    axes[0].set_ylabel('SSE')
    axes[0].set_title(f'SSE Distribution ({dataset_name})')
    axes[0].grid(axis='y', alpha=0.3)

    # 右图：折线图
    for algo in algos:
        sub = df[df['algorithm'] == algo]
        axes[1].plot(sub['run'], sub['sse'], marker='o', label=algo, alpha=0.7)
    axes[1].set_xlabel('Run')
    axes[1].set_ylabel('SSE')
    axes[1].set_title(f'SSE Across Runs ({dataset_name})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  图表已保存: {out_path}")


def main():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    k = 4  # 固定 k=4
    n_runs = 10

    # ── val 数据集 ──
    print("\n" + "="*60)
    print("val 数据集稳定性实验 (k=4, 10次运行)")
    print("="*60)

    X_val, _, _ = prepare(VAL_CONFIG)
    df_val = stability_experiment(X_val.values, k, n_runs)

    # 统计摘要
    summary_val = df_val.groupby('algorithm')['sse'].agg(['mean', 'std', 'min', 'max'])
    print("\nval 数据集统计摘要:")
    print(summary_val.to_string())

    df_val.to_csv(out_dir / "stability_val.csv", index=False, encoding='utf-8-sig')
    plot_stability(df_val, out_dir / "stability_val.png", "val")

    # ── train 数据集 ──
    print("\n" + "="*60)
    print("train 数据集稳定性实验 (k=4, 10次运行)")
    print("="*60)

    X_train, _, _ = prepare(TRAIN_CONFIG)
    df_train = stability_experiment(X_train.values, k, n_runs)

    summary_train = df_train.groupby('algorithm')['sse'].agg(['mean', 'std', 'min', 'max'])
    print("\ntrain 数据集统计摘要:")
    print(summary_train.to_string())

    df_train.to_csv(out_dir / "stability_train.csv", index=False, encoding='utf-8-sig')
    plot_stability(df_train, out_dir / "stability_train.png", "train")

    # ── 变异系数对比 ──
    print("\n" + "="*60)
    print("变异系数对比 (CV = std/mean)")
    print("="*60)

    cv_val = summary_val['std'] / summary_val['mean']
    cv_train = summary_train['std'] / summary_train['mean']

    cv_df = pd.DataFrame({
        'val_CV': cv_val,
        'train_CV': cv_train
    })
    print(cv_df.to_string())
    cv_df.to_csv(out_dir / "stability_cv.csv", encoding='utf-8-sig')

    print(f"\n所有结果已保存到: {out_dir}/")


if __name__ == "__main__":
    main()
