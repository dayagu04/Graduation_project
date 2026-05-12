"""
生成 k=4 聚类结果并分析群体画像
用于第五章的群体特征解读和可视化
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import TRAIN_CONFIG, prepare
from sklearn.cluster import KMeans


def analyze_clusters(X, labels, feature_names, out_dir: Path):
    """分析每个簇的特征画像"""
    n_clusters = len(np.unique(labels))

    # 计算每个簇的中心点（归一化后的值）
    centers = []
    for i in range(n_clusters):
        mask = labels == i
        centers.append(X[mask].mean(axis=0))
    centers = np.array(centers)

    # 保存聚类结果
    result_df = pd.DataFrame(X, columns=feature_names)
    result_df['cluster'] = labels
    result_df.to_csv(out_dir / "clustering_result_k4.csv", index=False, encoding='utf-8-sig')

    # 保存簇中心
    center_df = pd.DataFrame(centers, columns=feature_names)
    center_df.index.name = 'cluster'
    center_df.to_csv(out_dir / "cluster_centers_k4.csv", encoding='utf-8-sig')

    # 统计每个簇的样本数
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    print("\n=== 簇大小统计 ===")
    for i, size in cluster_sizes.items():
        print(f"簇 {i}: {size} 个样本 ({size/len(labels)*100:.1f}%)")

    # 打印每个簇的特征均值（归一化后）
    print("\n=== 簇中心特征（归一化值，0-1） ===")
    print(center_df.round(3).to_string())

    return centers, cluster_sizes


def plot_radar(centers, feature_names, cluster_sizes, out_dir: Path):
    """绘制雷达图"""
    # 选择最重要的 8 个特征（避免雷达图过于拥挤）
    important_features = [
        'StudyHours', 'Attendance', 'ExamScore', 'FinalGrade',
        'AssignmentCompletion', 'academic_composite', 'study_efficiency', 'Motivation'
    ]

    indices = [feature_names.index(f) for f in important_features if f in feature_names]
    selected_features = [feature_names[i] for i in indices]
    selected_centers = centers[:, indices]

    n_clusters = len(centers)
    angles = np.linspace(0, 2 * np.pi, len(selected_features), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(n_clusters):
        ax = axes[i]
        values = selected_centers[i].tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=f'簇 {i}')
        ax.fill(angles, values, alpha=0.25, color=colors[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(selected_features, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f'簇 {i} 特征画像 (n={cluster_sizes[i]})', fontsize=12, pad=20)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "cluster_radar_charts.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n雷达图已保存: {out_dir / 'cluster_radar_charts.png'}")


def plot_heatmap(centers, feature_names, out_dir: Path):
    """绘制热力图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(centers, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(centers)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f'簇 {i}' for i in range(len(centers))], fontsize=11)

    # 在每个格子中标注数值
    for i in range(len(centers)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{centers[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('聚类中心特征热力图（归一化值）', fontsize=14, pad=15)
    fig.colorbar(im, ax=ax, label='特征值 (0-1)')

    plt.tight_layout()
    plt.savefig(out_dir / "cluster_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {out_dir / 'cluster_heatmap.png'}")


def interpret_clusters(centers, feature_names, cluster_sizes):
    """生成群体画像文本描述"""
    interpretations = []

    for i in range(len(centers)):
        center = centers[i]
        size = cluster_sizes[i]

        # 提取关键特征
        feat_dict = {name: center[j] for j, name in enumerate(feature_names)}

        # 根据特征值生成描述
        desc = f"\n### 簇 {i}：{size} 人 ({size/cluster_sizes.sum()*100:.1f}%)\n\n"

        # 学习投入
        study_hours = feat_dict.get('StudyHours', 0)
        attendance = feat_dict.get('Attendance', 0)
        desc += f"**学习投入**：学习时间 {study_hours:.2f}，出勤率 {attendance:.2f}\n"

        # 学业表现
        exam = feat_dict.get('ExamScore', 0)
        final = feat_dict.get('FinalGrade', 0)
        assignment = feat_dict.get('AssignmentCompletion', 0)
        desc += f"**学业表现**：考试成绩 {exam:.2f}，最终成绩 {final:.2f}，作业完成率 {assignment:.2f}\n"

        # 学习效率
        efficiency = feat_dict.get('study_efficiency', 0)
        composite = feat_dict.get('academic_composite', 0)
        desc += f"**综合指标**：学习效率 {efficiency:.2f}，综合得分 {composite:.2f}\n"

        # 定性描述
        if exam > 0.7 and attendance > 0.7:
            desc += "\n**群体特征**：高成绩高出勤型，学习态度端正，成绩优异\n"
        elif exam > 0.5 and efficiency > 0.5:
            desc += "\n**群体特征**：高效学习型，学习效率高，投入产出比好\n"
        elif attendance > 0.6 and exam < 0.4:
            desc += "\n**群体特征**：努力但成绩欠佳型，出勤率高但成绩不理想，可能存在学习方法问题\n"
        else:
            desc += "\n**群体特征**：待提升型，学习投入和成绩均有较大提升空间\n"

        interpretations.append(desc)

    return "\n".join(interpretations)


def main():
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    print("="*60)
    print("生成 k=4 聚类结果并分析群体画像")
    print("="*60)

    # 加载数据
    X, feature_names, scaler = prepare(TRAIN_CONFIG)
    print(f"\n数据集: {len(X)} 样本, {len(feature_names)} 特征")

    # K-Means++ 聚类
    print("\n运行 K-Means++ (k=4)...")
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=20, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(X.values)

    # 分析簇
    centers, cluster_sizes = analyze_clusters(X.values, labels, feature_names, out_dir)

    # 可视化
    print("\n生成可视化...")
    plot_radar(centers, feature_names, cluster_sizes, out_dir)
    plot_heatmap(centers, feature_names, out_dir)

    # 生成文本描述
    print("\n生成群体画像描述...")
    interpretation = interpret_clusters(centers, feature_names, cluster_sizes)

    with open(out_dir / "cluster_interpretation.txt", "w", encoding='utf-8') as f:
        f.write(interpretation)

    print(interpretation)
    print(f"\n所有结果已保存到: {out_dir}/")


if __name__ == "__main__":
    main()
