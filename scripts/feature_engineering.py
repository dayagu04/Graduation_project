"""
通用特征工程模块
- 支持任意 CSV，不硬编码列名
- 通过 DatasetConfig 描述数据集结构
- 输出归一化后的 DataFrame 供聚类使用
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── 数据集配置 ─────────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    """描述一个数据集的结构，不含任何业务逻辑。"""
    path: str
    # 明确指定要使用的列（为空则自动选取所有数值列）
    feature_cols: List[str] = field(default_factory=list)
    # 自动检测模式下跳过的列（id、名字等）
    skip_cols: List[str] = field(default_factory=list)
    # 缺失值策略：'drop' | 'mean' | 'median' | 'zero'
    fill_strategy: str = "mean"
    separator: Optional[str] = None  # None = 自动检测


# 两个数据集的默认配置
TRAIN_CONFIG = DatasetConfig(
    path="data/train-data.csv",
    # 连续特征 + 衍生特征，排除二值/低基数分类特征和目标变量
    feature_cols=["StudyHours", "Attendance", "Age", "OnlineCourses",
                  "AssignmentCompletion", "ExamScore",
                  "academic_composite", "study_efficiency"],
)

VAL_CONFIG = DatasetConfig(
    path="data/val-data.csv",
    # assignments_submitted 整列全为空，直接跳过；student_id/name/gender 非数值
    skip_cols=["student_id", "name", "gender", "assignments_submitted"],
    fill_strategy="mean",
)


# ── 核心处理函数 ────────────────────────────────────────────────────────────────

def _detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """按配置加载原始 CSV，不做任何变换。"""
    sep = cfg.separator or _detect_sep(cfg.path)
    return pd.read_csv(cfg.path, sep=sep)


def select_features(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """
    确定最终特征列：
    1. 若 cfg.feature_cols 非空 → 直接使用
    2. 否则 → 自动选取数值列，排除 cfg.skip_cols
    """
    if cfg.feature_cols:
        missing = [c for c in cfg.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"指定列不存在: {missing}")
        return df[cfg.feature_cols].copy()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    selected = [c for c in num_cols if c not in cfg.skip_cols]
    return df[selected].copy()


def fill_missing(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """统一缺失值填充。"""
    if strategy == "drop":
        return df.dropna()
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    if strategy == "zero":
        return df.fillna(0)
    raise ValueError(f"Unknown fill_strategy: {strategy}")


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max 归一化，返回归一化后的 DataFrame 和 scaler（供反归一化）。"""
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df.values.astype(float))
    return pd.DataFrame(arr, columns=df.columns, index=df.index), scaler


# ── 衍生特征（按数据集结构判断是否能构造）──────────────────────────────────────

def _add_derived_train(df: pd.DataFrame) -> pd.DataFrame:
    """train-data 的衍生特征（若相关列存在则构造）。"""
    if {"ExamScore", "AssignmentCompletion", "Attendance"}.issubset(df.columns):
        # 学业综合得分：考试成绩占 60%，作业完成率占 40%
        df = df.copy()
        df["academic_composite"] = df["ExamScore"] * 0.6 + df["AssignmentCompletion"] * 0.4
    if {"StudyHours", "ExamScore"}.issubset(df.columns):
        df["study_efficiency"] = df["ExamScore"] / df["StudyHours"].clip(lower=1)
    return df


def _add_derived_val(df: pd.DataFrame) -> pd.DataFrame:
    """val-data 的衍生特征。"""
    if {"quiz1_marks", "quiz2_marks", "quiz3_marks"}.issubset(df.columns):
        df = df.copy()
        quiz_cols = ["quiz1_marks", "quiz2_marks", "quiz3_marks"]
        df["quiz_avg"]        = df[quiz_cols].mean(axis=1)
        df["quiz_volatility"] = df[quiz_cols].std(axis=1)
        df["quiz_trend"]      = df["quiz3_marks"] - df["quiz1_marks"]
    if {"lectures_attended", "total_lectures"}.issubset(df.columns):
        df["attendance_rate"] = df["lectures_attended"] / df["total_lectures"].clip(lower=1)
    if {"labs_attended", "total_lab_sessions"}.issubset(df.columns):
        df["lab_rate"] = df["labs_attended"] / df["total_lab_sessions"].clip(lower=1)
    if {"assignments_submitted", "total_assignments"}.issubset(df.columns):
        df["assignment_rate"] = df["assignments_submitted"] / df["total_assignments"].clip(lower=1)
    return df


# ── 公开接口 ───────────────────────────────────────────────────────────────────

def prepare(
    cfg: DatasetConfig,
    add_derived: bool = True,
) -> tuple[pd.DataFrame, list[str], MinMaxScaler]:
    """
    端到端特征工程流水线。

    Returns
    -------
    X_scaled : 归一化后的特征 DataFrame
    feature_names : 最终使用的列名列表
    scaler : 已拟合的 MinMaxScaler
    """
    raw = load_raw(cfg)

    # 判断数据集类型并添加衍生特征
    if add_derived:
        if "ExamScore" in raw.columns:
            raw = _add_derived_train(raw)
        elif "quiz1_marks" in raw.columns:
            raw = _add_derived_val(raw)

    feat_df = select_features(raw, cfg)
    # 衍生特征构造后可能引入新 NaN（如 assignment_rate 除以全缺列），先填充再归一化
    feat_df = fill_missing(feat_df, cfg.fill_strategy)
    # 若某列归一化后全为 0（min==max），直接丢弃，避免无意义特征
    feat_df = feat_df.loc[:, feat_df.std() > 1e-9]
    X_scaled, scaler = normalize(feat_df)

    return X_scaled, list(X_scaled.columns), scaler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="特征工程预览工具")
    parser.add_argument("--dataset", choices=["train", "val"], default="train")
    parser.add_argument("--no-derived", action="store_true")
    args = parser.parse_args()

    cfg = TRAIN_CONFIG if args.dataset == "train" else VAL_CONFIG
    X, names, _ = prepare(cfg, add_derived=not args.no_derived)

    print(f"\n数据集: {cfg.path}")
    print(f"样本数: {len(X)}  特征数: {len(names)}")
    print(f"特征列: {names}")
    print("\n前 3 行（归一化后）:")
    print(X.head(3).to_string())
    print("\n统计摘要:")
    print(X.describe().T[["mean", "std", "min", "max"]].to_string())
