"""
第五章图 5-1：学生成绩聚类分析原型系统架构图
三层架构（自下而上）：
  ① 数据摄取与预处理层
  ② 核心分析引擎层
  ③ 解释与反馈应用层
纯 matplotlib 绘制，免 graphviz 二进制依赖。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ── 架构定义 ──────────────────────────────────────────────────────────────────

# 画布 16 × 12（宽 × 高），y 自下而上
CANVAS_W, CANVAS_H = 16.0, 12.0

# 三层背景（y_bottom, y_top, 填充色, 边框色, 标签, 标签色）
LAYERS = [
    (0.5,  3.5,  "#e8f1fa", "#1f4e79",
     "① 数据摄取与预处理层  (Data Ingestion & Preprocessing)",
     "#1f4e79"),
    (4.0,  7.5,  "#fdf2e1", "#b9770e",
     "② 核心分析引擎层  (Core Analytics Engine)",
     "#b9770e"),
    (8.0, 11.5,  "#eaf5ea", "#1e7e34",
     "③ 解释与反馈应用层  (Interpretation & Feedback)",
     "#1e7e34"),
]

# 节点：id -> (label, layer, x 中心, 颜色)
# 每层 4 个节点沿 x 轴水平排布
NODE_W, NODE_H = 2.8, 1.15

# x 位置（按 4 列均匀分布，左右各留边距）
XS = [2.5, 6.5, 10.5, 14.0]

NODES = {
    # 第一层
    "db":     ("教务数据库\n(CSV / 学生成绩库)",       0, XS[0], "#4f81bd"),
    "clean":  ("缺失值清洗\n(均值填充 / 剔除)",         0, XS[1], "#4f81bd"),
    "feat":   ("特征工程\n(衍生指标计算)",              0, XS[2], "#4f81bd"),
    "norm":   ("Min-Max 无量纲化\n(0–1 归一化)",        0, XS[3], "#4f81bd"),

    # 第二层
    "kopt":   ("最佳 K 值评估\n(肘部法 + 轮廓系数)",     1, XS[0], "#e67e22"),
    "pso":    ("PSO 全局寻优\n(粒子群适应度搜索)",        1, XS[1], "#e67e22"),
    "kmeans": ("K-Means 局部微调\n(迭代精细化)",         1, XS[2], "#e67e22"),
    "out":    ("输出聚类质心\n与样本标签",               1, XS[3], "#e67e22"),

    # 第三层
    "viz":    ("多维画像可视化\n(雷达图 / 热力图)",       2, XS[0], "#27ae60"),
    "warn":   ("学业预警触发\n(薄弱群体识别)",           2, XS[1], "#27ae60"),
    "policy": ("个性化教学\n干预策略输出",                2, XS[2], "#27ae60"),
    "report": ("反馈报告\n(教师 / 学生 / 管理端)",        2, XS[3], "#27ae60"),
}

# 数据流箭头
FLOW_EDGES = [
    # 第一层内部
    ("db", "clean"),
    ("clean", "feat"),
    ("feat", "norm"),
    # 层间：预处理 → 分析引擎
    ("norm", "kopt", "up"),
    # 第二层内部
    ("kopt", "pso"),
    ("pso", "kmeans"),
    ("kmeans", "out"),
    # 层间：分析引擎 → 应用层
    ("out", "viz", "up"),
    # 第三层内部
    ("viz", "warn"),
    ("warn", "policy"),
    ("policy", "report"),
]


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def draw_node(ax, label: str, x: float, y: float, color: str) -> None:
    box = FancyBboxPatch(
        (x - NODE_W / 2, y - NODE_H / 2),
        NODE_W, NODE_H,
        boxstyle="round,pad=0.06,rounding_size=0.18",
        linewidth=1.6, edgecolor="#333333",
        facecolor=color, alpha=0.92,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")


def draw_layer(ax, y_bot: float, y_top: float,
               fill: str, edge: str, title: str, title_color: str) -> None:
    rect = FancyBboxPatch(
        (0.5, y_bot), CANVAS_W - 1.0, y_top - y_bot,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        linewidth=2.0, edgecolor=edge,
        facecolor=fill, alpha=0.55,
    )
    ax.add_patch(rect)
    ax.text(
        0.9, y_top - 0.30, title,
        fontsize=17, color=title_color, fontweight="bold",
        ha="left", va="center",
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.6,
               style="-|>", mutation=18, curved=False):
    kwargs = dict(arrowstyle=style, mutation_scale=mutation,
                  linewidth=lw, color=color)
    if curved:
        kwargs["connectionstyle"] = "arc3,rad=0.2"
    arrow = FancyArrowPatch((x1, y1), (x2, y2), **kwargs)
    ax.add_patch(arrow)


def plot_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 9.0))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H + 0.6)
    ax.axis("off")

    # 1. 层背景
    for y_bot, y_top, fill, edge, title, tcolor in LAYERS:
        draw_layer(ax, y_bot, y_top, fill, edge, title, tcolor)

    # 计算每层节点的中心 y
    layer_y_center = {
        0: (LAYERS[0][0] + LAYERS[0][1]) / 2 - 0.2,
        1: (LAYERS[1][0] + LAYERS[1][1]) / 2 - 0.2,
        2: (LAYERS[2][0] + LAYERS[2][1]) / 2 - 0.2,
    }

    # 2. 节点
    node_pos = {}
    for nid, (label, layer, x, color) in NODES.items():
        y = layer_y_center[layer]
        draw_node(ax, label, x, y, color)
        node_pos[nid] = (x, y)

    # 3. 箭头：横向同层 + 纵向跨层
    for edge in FLOW_EDGES:
        src, dst = edge[0], edge[1]
        direction = edge[2] if len(edge) > 2 else "right"
        x1, y1 = node_pos[src]
        x2, y2 = node_pos[dst]

        if direction == "up":
            # 纵向跨层：从源节点顶端到目标节点底端
            sx = x1
            sy = y1 + NODE_H / 2
            ex = x1   # 垂直向上对齐
            ey = y2 - NODE_H / 2
            # 先画一条粗主干箭头
            draw_arrow(ax, sx, sy, ex, ey, color="#c0392b", lw=2.2, mutation=22)
            # 再加一个横向标注 "数据流"
            mid_y = (sy + ey) / 2
            ax.text(
                sx + 0.25, mid_y,
                "数据流 / Data Flow",
                fontsize=13, color="#c0392b",
                va="center", ha="left", fontweight="bold",
                rotation=90,
            )
        else:
            # 同层横向
            sx = x1 + NODE_W / 2
            ex = x2 - NODE_W / 2
            sy = ey = y1
            draw_arrow(ax, sx, sy, ex, ey, color="#333333", lw=1.5)

    # 4. 左侧层级序号箭头（显式表达"自下而上"）
    ax.annotate(
        "", xy=(0.22, 11.3), xytext=(0.22, 0.7),
        arrowprops=dict(arrowstyle="-|>", color="#555555",
                        linewidth=2.0, mutation_scale=22),
    )
    ax.text(0.08, 6.0, "数据 → 分析 → 应用",
            fontsize=14, color="#555555", fontweight="bold",
            rotation=90, ha="center", va="center")

    # 5. 标题
    ax.text(
        CANVAS_W / 2, CANVAS_H + 0.35,
        "学生成绩聚类分析原型系统架构图",
        fontsize=22, fontweight="bold", ha="center", va="center",
        color="#222222",
    )

    # 6. 图例（右下角）
    legend_handles = [
        mpatches.Patch(facecolor="#4f81bd", edgecolor="#333", label="数据预处理模块"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="#333", label="核心分析模块"),
        mpatches.Patch(facecolor="#27ae60", edgecolor="#333", label="应用反馈模块"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right", bbox_to_anchor=(0.985, 0.005),
        fontsize=13, frameon=True, framealpha=0.92,
        title="模块类别", title_fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "5_系统应用_原型架构图.png"
    print(f"正在生成系统架构图 → {out_path} ...")
    plot_architecture(out_path)
    print(f"[OK] 已保存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
