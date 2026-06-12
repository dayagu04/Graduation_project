# 基于聚类算法的学生成绩分析与研究

毕业论文项目：使用 K-Means / K-Means++ / PSO-KMeans 对学生学业数据进行聚类分析，构建群体画像并提供学业诊断。

## 项目结构

```
Graduation_project/
├── data/               # 数据集
│   ├── train-data.csv      # 训练集（14,003 条，16 特征）
│   └── val-data.csv        # 验证集（300 条）
├── src/                # C++ 聚类算法实现
│   ├── main.cpp            # 主程序（CLI）
│   ├── kmeans.h/cpp        # K-Means++ 实现
│   ├── pso_kmeans.h/cpp    # PSO-KMeans 实现
│   └── csv_reader.h/cpp    # CSV 读取工具
├── scripts/            # Python 实验与可视化脚本
├── output/             # 算法中间输出（CSV/PNG）
├── results/            # 论文最终图表（300 DPI PNG）
├── md/                 # 论文文档（git submodule）
└── 论文_images/         # 从论文 docx 提取的插图
```

## 环境要求

- Python 3.9+
- GCC / G++ (MinGW, C++17) — 用于 C++ 实现
- 依赖：`pip install -r requirements.txt`

## 快速开始

### C++ 聚类算法

```bash
g++ -std=c++17 -Isrc src/main.cpp src/csv_reader.cpp src/kmeans.cpp src/pso_kmeans.cpp -o main.exe
./main.exe --data data/train-data.csv --algo pso-kmeans --k 4
```

可选参数见 `./main.exe --help`。

### Python 实验脚本

```bash
# 生成论文图表
python scripts/relabel_charts.py
python scripts/chapter45_viz.py
python scripts/regen_figs.py
```

## 数据集

特征列：StudyHours, Attendance, Resources, Extracurricular, Motivation, Internet, Gender, Age, LearningStyle, OnlineCourses, Discussions, AssignmentCompletion, ExamScore, EduTech, StressLevel, FinalGrade

## 算法性能（k=4，训练集）

| 算法 | SSE | Silhouette | 运行时间 |
|------|-----|-----------|---------|
| K-Means | 24272.67 | 0.1127 | 0.060s |
| K-Means++ | 24159.62 | 0.0894 | 0.135s |
| PSO-KMeans | 24538.18 | 0.0939 | 0.924s |

## 论文文档

论文相关文档位于 `md/` 子模块（独立仓库）。
