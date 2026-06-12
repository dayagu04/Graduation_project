"""
为 results/index.html 注入图片元数据 (FIG_DATA) 和交互 JS。
"""
import json
from pathlib import Path

INDEX_HTML = Path("results/index.html")


# ── 1. 章节定义 ────────────────────────────────────────────────────────────────

SECTIONS = [
    {
        "ch": "CHAPTER 03",
        "title": "数据预处理与特征工程构建",
        "desc": "基于训练集（14,003 条）与验证集（300 条），构建 16 原始 + 2 衍生 = 18 维特征空间，完成缺失值处理与 Min-Max 归一化。",
        "figures": ["fig3-1", "fig3-2", "fig3-3", "fig3-4", "fig3-5"],
    },
    {
        "ch": "CHAPTER 04",
        "title": "改进聚类算法模型构建与评估",
        "desc": "设计 PSO-KMeans 融合算法，与传统 K-Means、K-Means++ 在 SSE / 轮廓系数 / Davies-Bouldin / Calinski-Harabasz 四项指标对比，并通过 10 次独立运行验证统计稳定性。",
        "figures": ["fig4-1", "fig4-2", "fig4-3", "fig4-4", "fig4-5", "fig4-6", "fig4-7", "fig4-8"],
    },
    {
        "ch": "CHAPTER 05",
        "title": "聚类分析系统应用与学业诊断",
        "desc": "基于 K-Means++ (k=4) 的最终聚类结果，构建原型系统并完成 4 个学生群体的教育学画像解读与针对性教学干预策略。",
        "figures": ["fig5-1", "fig5-2", "fig5-3"],
    },
    {
        "ch": "APPENDIX",
        "title": "综合总结与补充图",
        "desc": "实验综合面板与对比表格、补充画像图，可作为答辩 PPT 的高密度索引页使用。",
        "figures": ["sum-1", "sum-2", "sup-1", "sup-2"],
    },
]


# ── 2. 每张图的元数据 ───────────────────────────────────────────────────────────

FIGS = {
    "fig3-1": {
        "tag": "图 3-1", "tagClass": "thesis",
        "file": "1_特征工程_KDE分布检验.png",
        "title": "关键特征核密度估计（KDE）分布",
        "subtitle": "ExamScore / StudyHours / Attendance / StressLevel",
        "purpose": "为基于欧氏距离的 K-Means 算法的合理性提供统计学证据：考试成绩近似高斯分布，符合 SSE 最小化的几何假设。",
        "dataSource": [("数据来源", "data/train-data.csv"), ("样本", "14,003 条"),
                       ("特征", "ExamScore / StudyHours / Attendance / StressLevel")],
        "method": "对四个关键特征绘制核密度估计曲线（Seaborn kdeplot, fill=True），并在子图右上角标注偏度（Skewness）与峰度（Kurtosis）数值。",
        "findings": [
            "(a) ExamScore 呈良好的高斯分布形态（|Skew|≈0），为欧氏距离提供坚实假设",
            "(b)(c) StudyHours、Attendance 呈轻度偏态分布",
            "(d) StressLevel 为离散型多峰分布，刻画三档压力等级",
        ],
        "insight": "数据空间整体呈凸集形状分布，K-Means 通过最小化 SSE 在该分布上能有效收敛。",
        "code": "scripts/feature_viz.py · plot_kde()",
    },
    "fig3-2": {
        "tag": "图 3-2", "tagClass": "thesis",
        "file": "1_特征工程_全局箱线图.png",
        "title": "Min-Max 归一化后全局箱线图",
        "subtitle": "16 个数值特征的离群点分布",
        "purpose": "直观展示即使 Min-Max 归一化后仍存在极端学习行为（如 Internet 7.45%、StudyHours 0.32% 离群点），反向论证后续 PSO 全局优化提升鲁棒性的必要性。",
        "dataSource": [("数据来源", "data/train-data.csv"), ("处理", "MinMaxScaler 缩放至 [0,1]"),
                       ("离群点检测", "IQR 准则（>1.5×IQR）")],
        "method": "对所有数值特征执行 Min-Max 归一化，绘制全局 Boxplot，红点突显离群点，并在每个箱子上方标注离群点数量。",
        "findings": [
            "Internet：1043 个离群点（7.45%）— 二值特征极端取值",
            "StudyHours：45 个离群点（0.32%）— 极端高学习时长样本",
            "其余连续型特征 IQR 离群点为零，分布整体均匀",
        ],
        "insight": "极端样本的存在使随机初始化敏感的 K-Means 易陷入局部最优，验证 PSO 全局搜索的工程价值。",
        "code": "scripts/feature_viz.py · plot_global_boxplot()",
    },
    "fig3-3": {
        "tag": "图 3-3", "tagClass": "thesis",
        "file": "1_特征工程_衍生特征分布.png",
        "title": "面向教育场景的衍生特征分布",
        "subtitle": "成绩波动率 / 趋势 / 综合得分 / 学习效率",
        "purpose": "验证衍生特征构造的合理性，论证其相比原始单一指标更能刻画学生学业发展的动态轨迹。",
        "dataSource": [("(a)(b)", "val-data.csv 衍生：quiz_volatility / quiz_trend"),
                       ("(c)(d)", "train-data.csv 衍生：academic_composite / study_efficiency")],
        "method": "对四个衍生特征分别绘制直方图叠加 KDE 曲线，标注偏度峰度。学业综合得分公式：0.6·ExamScore + 0.4·AssignmentCompletion；学习效率：ExamScore / max(StudyHours,1)。",
        "findings": [
            "(a) quiz_volatility 高度右偏，揭示稳定型与波动型学生的明显分化",
            "(b) quiz_trend 接近正态，正负值刻画学习进步/退步方向",
            "(c) academic_composite 呈双峰，对应学业核心分群",
            "(d) study_efficiency 长尾分布，少数样本学习效率显著偏离均值",
        ],
        "insight": "衍生特征显著扩展了聚类的可识别维度，为后续 \"考试强但作业弱\" 等细分群体识别提供数据基础。",
        "code": "scripts/feature_engineering.py · _add_derived_train()",
    },
    "fig3-4": {
        "tag": "图 3-4", "tagClass": "thesis",
        "file": "1_特征工程_特征对比.png",
        "title": "原始特征 vs 衍生特征二维分布对比",
        "subtitle": "成绩二维 vs 波动率-趋势二维",
        "purpose": "对比原始成绩特征空间与衍生特征空间的聚类可分性，论证特征工程的工程价值。",
        "dataSource": [("数据集", "train-data.csv & val-data.csv")],
        "method": "左图绘制原始特征二维散点（如 ExamScore × FinalGrade），右图绘制衍生特征二维散点（quiz_volatility × quiz_trend），同一坐标轴范围便于直接对比。",
        "findings": [
            "原始特征空间：样本沿对角线密集分布，群体边界模糊",
            "衍生特征空间：样本在二维平面上形成更明显的群体边界",
        ],
        "insight": "衍生特征的方差更大、分布更扩散，为聚类算法提供了更显著的群体可分性。",
        "code": "scripts/visualize.py",
    },
    "fig3-5": {
        "tag": "图 3-5", "tagClass": "thesis",
        "file": "1_特征工程_半三角相关性矩阵.png",
        "title": "训练集 18 维特征 Pearson 相关性矩阵",
        "subtitle": "下三角 RdBu_r 学术标准画法",
        "purpose": "辅助分析特征独立性、衍生特征与原始特征的相关性，为特征筛选提供依据。",
        "dataSource": [("特征", "16 原始 + 2 衍生 = 18 维"),
                       ("方法", "Pearson 相关系数 + np.triu mask")],
        "method": "对 18 维归一化特征计算 Pearson 相关系数矩阵，使用 np.triu 生成上三角掩码仅显示下三角，配色采用 RdBu_r 冷暖对比，相关系数保留两位小数。",
        "findings": [
            "academic_composite 与 ExamScore（r≈0.78）、AssignmentCompletion（r≈0.62）强相关 — 验证加权公式正确性",
            "study_efficiency 与 ExamScore 中等正相关、与 StudyHours 弱负相关 — 反映高效学习者特征",
            "其余原始特征间相关性普遍较低（|r|<0.3）— 大部分特征相互独立",
        ],
        "insight": "衍生特征与其源特征的强相关属预期，整体特征空间冗余度低，无需进一步降维即可输入聚类。",
        "code": "scripts/feature_viz.py · plot_corr_heatmap()",
    },
    "fig4-1": {
        "tag": "图 4-1", "tagClass": "thesis",
        "file": "2_算法对比_PSOKMeans算法流程图.png",
        "title": "PSO-KMeans 融合算法执行流程图",
        "subtitle": "Exploration（PSO 全局）+ Exploitation（K-Means 局部）",
        "purpose": "可视化展示 PSO-KMeans 的两阶段融合架构与算法执行逻辑。",
        "dataSource": [("阶段 1", "PSO：N=20 粒子，T=30 代"),
                       ("阶段 2", "K-Means：max_iter=100")],
        "method": "11 节点完整流程图（开始/结束椭圆 / 处理矩形 / 判定菱形），含\"否\"回环箭头与阶段分色（PSO 蓝、K-Means 紫）。",
        "findings": [
            "PSO 阶段：粒子群搜索 → 适应度 SSE → pbest/gbest 更新 → 速度位置更新 → 迭代判定",
            "K-Means 阶段：以 gbest 为初始质心 → 局部精细化 → 输出最终标签",
        ],
        "insight": "通过 PSO 的全局探索保证收敛到高质量初始解，再用 K-Means 局部开发实现精细化收敛，避免单一算法陷入局部最优。",
        "code": "scripts/pso_kmeans.py · PSOKMeans.fit()",
    },
    "fig4-2": {
        "tag": "图 4-2", "tagClass": "thesis",
        "file": "2_算法对比_K值评估.png",
        "title": "肘部法 + 轮廓系数联合 K 值评估",
        "subtitle": "(a) Elbow Method  (b) Silhouette Score",
        "purpose": "确定最佳聚类数 k，平衡聚类质量与教育学可解释性。",
        "dataSource": [("范围", "k = 2..8"), ("算法", "K-Means++"),
                       ("数据", "train-data.csv，14,003 条")],
        "method": "对 k=2..8 逐一拟合 K-Means++，记录 SSE 与 Silhouette，并在 k=4 处加红色虚线标注选定值。",
        "findings": [
            "k=3→4：SSE 下降 1291.87",
            "k=4→5：SSE 下降 1280.76（衰减出现）",
            "k=5 时 Silhouette=0.1094，k=4 时为 0.0894",
        ],
        "insight": "综合 SSE 衰减拐点 + 轮廓系数 + 教育学可解释性，最终选定 k=4 — 既保证聚类质量，又匹配实际教育群体粒度。",
        "code": "scripts/cluster_experiments.py · evaluate_k_range()",
    },
    "fig4-3": {
        "tag": "图 4-3", "tagClass": "thesis",
        "file": "2_算法对比_K4轮廓系数分布图.png",
        "title": "K=4 轮廓系数刀刃分布图",
        "subtitle": "Silhouette Plot · 簇内紧密性诊断",
        "purpose": "诊断 K=4 时各簇内部的紧密性与样本归属合理性。",
        "dataSource": [("方法", "sklearn.metrics.silhouette_samples"),
                       ("采样", "5,000 / 14,003")],
        "method": "对每个样本计算轮廓系数，按簇分组并升序排列，水平填充形成刀刃形状；红色虚线标记全局平均值。",
        "findings": [
            "簇 0（男生课外活跃型）：均值 +0.1444，无负样本",
            "簇 3（女生课外活跃型）：均值 +0.1415，无负样本",
            "簇 1（混合均衡型）：均值 +0.0285，2.97% 负样本（过渡型）",
            "全局平均 = 0.0895",
        ],
        "insight": "所有簇均值为正，证明聚类结构合理；簇 1 是过渡型群体，正是教育技术推广干预的主要目标。",
        "code": "scripts/chapter4_extra_viz.py · plot_silhouette()",
    },
    "fig4-4": {
        "tag": "图 4-4", "tagClass": "thesis",
        "file": "2_算法对比_性能指标.png",
        "title": "三算法在不同评估指标下的性能对比",
        "subtitle": "(a) SSE  (b) Silhouette  (c) DB  (d) CH",
        "purpose": "全面对比 K-Means、K-Means++、PSO-KMeans 在 k=3,4,5 下的多维度性能。",
        "dataSource": [("k 值", "3, 4, 5"), ("种子", "42（可复现）")],
        "method": "对每个算法在每个 k 下计算 SSE、Silhouette、Davies-Bouldin、Calinski-Harabasz，绘制 2×2 折线对比图。",
        "findings": [
            "k=4 时 SSE：K-Means++ (24159.62) < K-Means (24272.67) < PSO-KMeans (24538.18)",
            "k=4 时 DB：K-Means++ (2.375) < PSO-KMeans (2.591) < K-Means (2.716)",
            "K-Means++ 在单次运行中综合最优",
        ],
        "insight": "单次运行 PSO-KMeans 不一定最优 — 真正的优势体现在多次运行的稳定性（见图 4-5）。",
        "code": "scripts/cluster_experiments.py · compare_algorithms()",
    },
    "fig4-5": {
        "tag": "图 4-5", "tagClass": "thesis",
        "file": "2_算法对比_稳定性.png",
        "title": "三算法 10 次独立运行 SSE 稳定性对比",
        "subtitle": "(a) 箱线图  (b) 跨运行波动折线",
        "purpose": "通过 10 次不同种子的独立运行，量化算法对随机初始化的敏感性，验证 PSO-KMeans 的统计稳定性。",
        "dataSource": [("种子范围", "42..51"), ("k", "4"), ("运行数", "10")],
        "method": "对三种算法各运行 10 次，统计 SSE 均值/标准差/变异系数 CV，绘制箱线图与折线图。",
        "findings": [
            "PSO-KMeans：均值 24,387.51（最低）、标准差 277.64（最低）、CV 1.14%",
            "K-Means++：均值 24,451.59、标准差 284.09（最高）、CV 1.16%",
            "K-Means：均值 24,598.58、标准差 281.43、CV 1.14%",
        ],
        "insight": "PSO-KMeans 通过全局搜索机制\"断崖式\"降低初始化随机性影响，是论文核心创新点的关键证据。",
        "code": "scripts/stability_experiment.py · stability_experiment()",
    },
    "fig4-6": {
        "tag": "图 4-6", "tagClass": "thesis",
        "file": "2_算法对比_PSO收敛曲线.png",
        "title": "PSO-KMeans 全局最优 SSE 收敛轨迹",
        "subtitle": "30 代迭代过程",
        "purpose": "展示 PSO 算法寻找全局最优解的内部收敛过程，验证 30 代迭代的充分性。",
        "dataSource": [("粒子数", "20"), ("最大迭代", "30"),
                       ("惯性权重", "0.729 → 0.4 线性递减")],
        "method": "在 fit() 中记录每代 global_best_fit，绘制折线图，标注初始化 SSE、PSO 收敛 SSE、K-Means 精细化后 SSE。",
        "findings": [
            "前 ~10 代快速收敛，下降占总幅度的绝大部分",
            "后 20 代进入精细搜索阶段",
            "末代 SSE 与 K-Means 精细化后 SSE 高度吻合",
        ],
        "insight": "30 代迭代足以让 PSO 收敛到良好区域，进一步增加迭代收益递减；在 14k 样本规模下单次运行约 0.9 秒，效率可接受。",
        "code": "scripts/pso_kmeans.py · convergence_history_",
    },
    "fig4-7": {
        "tag": "图 4-7", "tagClass": "thesis",
        "file": "3_聚类结果_PCA降维.png",
        "title": "基于 PCA 线性映射的聚类结果二维分布",
        "subtitle": "PC1 × PC2 散点图",
        "purpose": "在低维空间直观展示聚类结果的整体分布形态。",
        "dataSource": [("方法", "sklearn.decomposition.PCA(n_components=2)")],
        "method": "对 18 维归一化特征执行主成分分析，投影至 PC1-PC2 平面，按簇着色，红色 X 标记聚类中心。",
        "findings": [
            "四个簇在主成分空间呈大致分离的分布",
            "PC1 主要由 Extracurricular、EduTech 主导",
            "线性映射在多维特征交叉时存在一定信息损失",
        ],
        "insight": "PCA 给出整体分布的全局视角，但局部结构需借助 t-SNE 进一步验证。",
        "code": "scripts/visualize.py",
    },
    "fig4-8": {
        "tag": "图 4-8", "tagClass": "thesis",
        "file": "3_聚类结果_tSNE降维.png",
        "title": "基于 t-SNE 非线性流形的聚类结果分布",
        "subtitle": "局部结构保真展示",
        "purpose": "通过非线性降维展示局部一致性更高的聚类形态，弥补 PCA 在多维交叉时的局限。",
        "dataSource": [("方法", "sklearn.manifold.TSNE")],
        "method": "对 18 维归一化特征执行 t-SNE 非线性映射至二维平面，按簇着色。",
        "findings": [
            "四个学生群体在局部流形空间形成相互远离的聚集区域",
            "簇之间边界划分清晰，簇内样本距离紧密",
        ],
        "insight": "t-SNE 局部一致性更高，揭示出 PCA 中模糊的群体边界，进一步印证聚类结构的合理性。",
        "code": "scripts/visualize.py",
    },
    "fig5-1": {
        "tag": "图 5-1", "tagClass": "thesis",
        "file": "5_系统应用_原型架构图.png",
        "title": "学生成绩聚类分析原型系统架构图",
        "subtitle": "数据 → 分析 → 应用 三层架构",
        "purpose": "为论文第五章提供原型系统的整体工程视图，展示模块解耦与数据流向。",
        "dataSource": [("①", "数据摄取与预处理层"), ("②", "核心分析引擎层"),
                       ("③", "解释与反馈应用层")],
        "method": "三层背景分色（蓝/橙/绿），每层 4 个模块横向排布，纵向粗红色\"数据流\"箭头贯通三层，左侧整体方向箭头。",
        "findings": [
            "① 教务数据库 → 缺失值清洗 → 特征工程 → Min-Max 无量纲化",
            "② 最佳 K 值评估 → PSO 全局寻优 → K-Means 局部微调 → 输出质心标签",
            "③ 多维画像可视化 → 学业预警触发 → 个性化干预 → 反馈报告",
        ],
        "insight": "三层架构清晰划分了 ETL、算法、应用职责，为后续工程化部署奠定基础。",
        "code": "scripts/system_architecture.py",
    },
    "fig5-2": {
        "tag": "图 5-2", "tagClass": "thesis",
        "file": "4_群体画像_均值偏离分析图.png",
        "title": "K-Means++ (k=4) 各簇质心相对全局均值偏离图",
        "subtitle": "Diverging Bar Chart · 6 关键特征",
        "purpose": "用双向条形图揭示分群的真正驱动维度，破除\"成绩分层\"的直觉假设。",
        "dataSource": [("方法", "K-Means++ k=4"), ("基准", "训练集全局均值")],
        "method": "计算 4 簇质心 - 全局均值的偏离量，正向红色 / 负向蓝色，6 个关键特征水平排布。",
        "findings": [
            "(a) 簇 0：Extracurricular +0.406，EduTech +0.291，StressLevel +0.017",
            "(b) 簇 1：EduTech −0.709（完全不使用教育技术 — 最强分群信号）",
            "(c) 簇 2：Extracurricular −0.594（完全不参与课外活动）",
            "(d) 簇 3：Extracurricular +0.406、study_efficiency −0.003",
        ],
        "insight": "学业指标在四簇间几乎无差异（|偏离|<0.01），真正驱动聚类的是 Extracurricular + EduTech + StressLevel 三个行为维度。",
        "code": "scripts/relabel_charts.py · fig_deviation()",
    },
    "fig5-3": {
        "tag": "图 5-3", "tagClass": "thesis",
        "file": "4_群体画像_核心特征对比雷达图.png",
        "title": "K=4 各学生群体多维特征对比雷达图",
        "subtitle": "6 维特征 · 4 簇叠加",
        "purpose": "用雷达图直观对比四个群体在六个核心特征上的整体形态。",
        "dataSource": [("特征", "StudyHours / Extracurricular / EduTech / StressLevel / study_efficiency / academic_composite")],
        "method": "极坐标系叠加 4 条折线，红蓝绿紫四色 + alpha=0.15 半透明填充。",
        "findings": [
            "簇 0、簇 3 在 Extracurricular 与 EduTech 维度上达 1.0",
            "簇 2 在 Extracurricular 维度上为 0.0，形成强反差",
            "学习投入与成绩相关维度的雷达形状高度相似",
        ],
        "insight": "雷达图直观呼应图 5-2 的偏离分析，揭示分群本质是\"行为模式\"而非\"成绩水平\"。",
        "code": "scripts/radar_overlay.py",
    },
    "sum-1": {
        "tag": "总结", "tagClass": "append",
        "file": "5_实验总结报告.png",
        "title": "实验总结报告（8 面板）",
        "subtitle": "数据概览 + 算法对比 + 群体分布 + 关键结论",
        "purpose": "作为答辩 PPT 的高密度索引页，一图浓缩全部实验核心。",
        "dataSource": [("面板数", "8"), ("覆盖范围", "全章节核心指标")],
        "method": "GridSpec 多面板组合：数据卡片 / 肘部法 / SSE 柱状 / 簇大小饼图 / 稳定性箱线 / CV 柱状 / 综合质量雷达 / 文字结论。",
        "findings": [
            "k = 4（综合三指标 + 教育学可解释性）",
            "PSO-KMeans 均值最低 24,387.51、标准差 277.64",
            "分群主因：Extracurricular / EduTech / StressLevel",
        ],
        "insight": "高密度面板适合 1 分钟介绍全实验，便于答辩开场使用。",
        "code": "scripts/summary_panels.py · fig_experiment_summary()",
    },
    "sum-2": {
        "tag": "对比表", "tagClass": "append",
        "file": "2_算法对比_综合表格.png",
        "title": "三算法综合性能对比表",
        "subtitle": "k=3,4,5 多指标 + 稳定性",
        "purpose": "数值化对比表，便于论文正文引用与答辩问答。",
        "dataSource": [("上表", "SSE / Silhouette / DB / Time @ k=3,4,5"),
                       ("下表", "10 次运行稳定性 (Mean / Std / CV)")],
        "method": "matplotlib.table 双表叠加，指标分行分色，PSO-KMeans 行高亮。",
        "findings": [
            "K-Means++ 单次最优（k=4 SSE=24159.62）",
            "PSO-KMeans 多次最稳（CV=1.14%）",
        ],
        "insight": "通过单次 vs 多次的对比，揭示选择 PSO-KMeans 是基于统计意义而非单次结果。",
        "code": "scripts/summary_panels.py · fig_compare_table()",
    },
    "sup-1": {
        "tag": "补充", "tagClass": "append",
        "file": "4_群体画像_热力图.png",
        "title": "聚类中心特征热力图",
        "subtitle": "全部 18 维特征详细对照",
        "purpose": "提供 18 维特征上四个簇质心的精细数值对照，是图 5-2 / 5-3 的细节补充。",
        "dataSource": [("数据", "output/cluster_centers_k4.csv")],
        "method": "imshow 热力图，YlOrRd 配色，单元格内标注归一化值（保留两位小数）。",
        "findings": [
            "高度一致维度：StudyHours / Attendance / ExamScore / FinalGrade",
            "显著分化维度：Extracurricular / Gender / EduTech",
            "中等差异维度：Motivation / StressLevel / Discussions",
        ],
        "insight": "热力图揭示了在论文中未细论的次要维度差异，可作为附录引用。",
        "code": "scripts/generate_cluster_profiles.py · plot_heatmap()",
    },
    "sup-2": {
        "tag": "补充", "tagClass": "append",
        "file": "4_学业诊断_报告.png",
        "title": "自动生成的学业诊断报告",
        "subtitle": "结构化文本：群体标签 + 画像 + 干预建议",
        "purpose": "展示原型系统的应用层输出，对接教务管理流程。",
        "dataSource": [("规则", "scripts/visualize.py 中的 rule-based 分类器")],
        "method": "基于聚类质心的规则化匹配生成文本卡片，包含群体标签、画像描述、教学建议三段式结构。",
        "findings": [
            "簇 0：时间管理 + 压力管理（一对一辅导 / 心理讲座）",
            "簇 1：教育技术接受度低（在线学习推广）",
            "簇 2：缺乏课外活动参与（社团引导）",
            "簇 3：学习效率低（学业导师制）",
        ],
        "insight": "完成\"学业诊断 → 教学建议\"的最终闭环，是系统应用价值的直接证明。",
        "code": "scripts/visualize.py",
    },
}


# ── 3. 生成 HTML 主网格 ─────────────────────────────────────────────────────────

def render_grid_html() -> str:
    out = []
    for sec in SECTIONS:
        out.append(f'<div class="section">')
        out.append(f'  <h2 class="section-title"><span>{sec["title"]}</span><span class="ch">{sec["ch"]}</span></h2>')
        out.append(f'  <p class="section-desc">{sec["desc"]}</p>')
        out.append(f'  <div class="image-grid">')

        for fid in sec["figures"]:
            f = FIGS[fid]
            out.append(f'    <div class="image-card" data-id="{fid}">')
            out.append(f'      <div class="thumb">')
            out.append(f'        <span class="wheel-hint">滚轮 ↓ 看详情</span>')
            out.append(f'        <img src="{f["file"]}" alt="{f["title"]}" loading="lazy">')
            out.append(f'        <div class="thumb-overlay"><div class="actions">'
                       f'<span>🔍 点击放大</span><span>滚轮展开详情</span></div></div>')
            out.append(f'      </div>')
            out.append(f'      <div class="caption">')
            out.append(f'        <div class="tag-row"><span class="figure-tag {f["tagClass"]}">{f["tag"]}</span></div>')
            out.append(f'        <h3>{f["title"]}</h3>')
            out.append(f'        <p>{f["subtitle"]}</p>')
            out.append(f'      </div>')
            out.append(f'    </div>')

        out.append(f'  </div>')
        out.append(f'</div>')
    return "\n".join(out)


# ── 4. 应用层 JS ────────────────────────────────────────────────────────────────

APP_JS = r"""
(function () {
  const FIG_DATA = JSON.parse(document.getElementById('figure-data').textContent);
  const cards = Array.from(document.querySelectorAll('.image-card'));
  const allIds = cards.map(c => c.dataset.id);

  // Lightbox elements
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lb-img');
  const lbCap = document.getElementById('lb-caption');
  const lbClose = document.getElementById('lb-close');
  const lbPrev = document.getElementById('lb-prev');
  const lbNext = document.getElementById('lb-next');

  // Drawer elements
  const drawer = document.getElementById('drawer');
  const backdrop = document.getElementById('backdrop');
  const dClose = document.getElementById('drawer-close');
  const dFig = document.getElementById('d-fig');
  const dTitle = document.getElementById('d-title');
  const dSub = document.getElementById('d-sub');
  const dBody = document.getElementById('d-body');

  let lbIndex = 0;
  let drawerOpen = false;

  // ── Lightbox
  function openLightbox(id) {
    const f = FIG_DATA[id];
    if (!f) return;
    lbIndex = allIds.indexOf(id);
    lbImg.src = f.file;
    lbImg.alt = f.title;
    lbCap.textContent = `${f.tag} · ${f.title}`;
    lb.classList.add('visible');
    lb.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }
  function closeLightbox() {
    lb.classList.remove('visible');
    lb.setAttribute('aria-hidden', 'true');
    if (!drawerOpen) document.body.style.overflow = '';
  }
  function navLightbox(delta) {
    lbIndex = (lbIndex + delta + allIds.length) % allIds.length;
    openLightbox(allIds[lbIndex]);
  }

  // ── Drawer
  function openDrawer(id) {
    const f = FIG_DATA[id];
    if (!f) return;
    dFig.textContent = f.tag.toUpperCase();
    dTitle.textContent = f.title;
    dSub.textContent = f.subtitle || '';

    let html = '';
    if (f.purpose) {
      html += `<section><h3>研究目的</h3><p>${f.purpose}</p></section>`;
    }
    if (f.dataSource && f.dataSource.length) {
      html += '<section><h3>数据来源 / 参数</h3><dl class="kv">';
      f.dataSource.forEach(([k, v]) => {
        html += `<dt>${k}</dt><dd>${v}</dd>`;
      });
      html += '</dl></section>';
    }
    if (f.method) {
      html += `<section><h3>方法</h3><p>${f.method}</p></section>`;
    }
    if (f.findings && f.findings.length) {
      html += '<section><h3>关键发现</h3><ul>';
      f.findings.forEach(item => { html += `<li>${item}</li>`; });
      html += '</ul></section>';
    }
    if (f.insight) {
      html += `<section><h3>结论 / 启示</h3><p>${f.insight}</p></section>`;
    }
    if (f.code) {
      html += `<section><h3>对应代码</h3><pre>${f.code}</pre></section>`;
    }
    dBody.innerHTML = html;

    drawer.classList.add('visible');
    backdrop.classList.add('visible');
    drawer.setAttribute('aria-hidden', 'false');
    drawerOpen = true;
    document.body.style.overflow = 'hidden';
  }
  function closeDrawer() {
    drawer.classList.remove('visible');
    backdrop.classList.remove('visible');
    drawer.setAttribute('aria-hidden', 'true');
    drawerOpen = false;
    if (!lb.classList.contains('visible')) document.body.style.overflow = '';
  }

  // ── 卡片绑定
  cards.forEach(card => {
    const id = card.dataset.id;

    // 点击 → 放大
    card.addEventListener('click', e => {
      // 不阻止链接默认行为，但本卡片本身没链接
      openLightbox(id);
    });

    // 悬停 + 滚轮 → 打开详情抽屉
    let wheelLock = false;
    card.addEventListener('wheel', e => {
      // 仅向下滚轮触发；若已打开抽屉则忽略
      if (wheelLock || drawerOpen) return;
      if (e.deltaY > 0) {
        e.preventDefault();
        wheelLock = true;
        openDrawer(id);
        // 防抖：500ms 内不再触发
        setTimeout(() => { wheelLock = false; }, 500);
      }
    }, { passive: false });
  });

  // 关闭按钮
  lbClose.addEventListener('click', closeLightbox);
  lbPrev.addEventListener('click', e => { e.stopPropagation(); navLightbox(-1); });
  lbNext.addEventListener('click', e => { e.stopPropagation(); navLightbox(+1); });
  lb.addEventListener('click', e => { if (e.target === lb) closeLightbox(); });

  dClose.addEventListener('click', closeDrawer);
  backdrop.addEventListener('click', closeDrawer);

  // 键盘控制
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      if (drawerOpen) closeDrawer();
      else if (lb.classList.contains('visible')) closeLightbox();
    } else if (lb.classList.contains('visible')) {
      if (e.key === 'ArrowLeft') navLightbox(-1);
      else if (e.key === 'ArrowRight') navLightbox(+1);
    }
  });
})();
"""


# ── 5. 注入 ──────────────────────────────────────────────────────────────────

def main() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    # 主网格替换 <main id="grid-root"></main> 占位
    grid_html = render_grid_html()
    html = html.replace(
        '<main id="grid-root"></main>',
        f'<main id="grid-root">\n{grid_html}\n        </main>',
    )

    # 元数据 JSON 注入
    fig_json = json.dumps(FIGS, ensure_ascii=False)
    html = html.replace("__FIGURE_DATA__", fig_json)

    # JS 注入
    html = html.replace("__APP_JS__", APP_JS)

    INDEX_HTML.write_text(html, encoding="utf-8")
    print(f"[OK] 已注入 {len(FIGS)} 张图片元数据 + 交互 JS → {INDEX_HTML.resolve()}")


if __name__ == "__main__":
    main()
