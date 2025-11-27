# eval_tabpfn.py
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier

import hydra
from omegaconf import DictConfig, OmegaConf

# =========================
# 工具函数
# =========================

TEXT_LENGTH_DROP_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 200
N_ENSEMBLE_CONFIGURATIONS = 16

def load_data(cfg: DictConfig):
    """
    (重构版 - 基于 field_lengths 自动判断列类型)
    """
    target = cfg.target
    print(f"[INFO] 正在加载 target: {target} (自动推断列类型)")

    try:
        # --- 1. 加载数据 ---
        X_train_full = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        y_train_tensor = torch.load(cfg.labels_train_eval_tabular)
        y_train_full = y_train_tensor.numpy()

        X_test_full = pd.read_csv(cfg.data_test_eval_tabular, header=None)
        y_test_tensor = torch.load(cfg.labels_test_eval_tabular)
        y_test_full = y_test_tensor.numpy()

        # --- 2. 加载 field_lengths 并计算索引 ---
        # 假设 field_lengths_tabular 是一个 .pt 文件 (Torch Tensor) 或 .npy
        # 如果是其他格式 (如json)，请根据实际情况调整
        field_lengths_path = cfg.field_lengths_tabular
        print(f"[INFO] 读取字段长度文件: {field_lengths_path}")
        
        try:
            # 尝试作为 torch tensor 加载
            field_lengths = torch.load(field_lengths_path)
            if isinstance(field_lengths, torch.Tensor):
                field_lengths = field_lengths.numpy()
        except Exception:
            # 回退：尝试作为 numpy 加载
            field_lengths = np.load(field_lengths_path)
        
        # 展平以防万一
        field_lengths = np.array(field_lengths).flatten()
        
        # 校验列数是否匹配
        n_cols_data = X_train_full.shape[1]
        n_cols_lengths = len(field_lengths)
        if n_cols_data != n_cols_lengths:
            print(f"🔴 错误：CSV 列数 ({n_cols_data}) 与 field_lengths 长度 ({n_cols_lengths}) 不匹配！")
            sys.exit(1)

        # === 核心修改逻辑 ===
        # TIP 假设：field_len == 1 -> 连续特征； >1 -> 类别特征
        con_indices = [i for i, fl in enumerate(field_lengths) if fl == 1]
        cat_indices = [i for i, fl in enumerate(field_lengths) if fl > 1]
        
        print(f"[INFO] 自动检测结果:")
        print(f"      - 数值列数量: {len(con_indices)}")
        print(f"      - 类别列数量: {len(cat_indices)}")

        # --- 3. 定义列名 ---
        # 给所有列一个通用名字，方便后续按名字索引
        all_col_names = [f"col_{i}" for i in range(n_cols_data)]
        X_train_full.columns = all_col_names
        X_test_full.columns = all_col_names

        # 根据索引提取对应的列名列表
        num_cols = [all_col_names[i] for i in con_indices]
        cat_cols = [all_col_names[i] for i in cat_indices]

        # --- 4. 标签处理 (1-indexed -> 0-indexed) ---
        if cfg.task == 'classification':
            label_min = np.min(y_train_full)
            label_max = np.max(y_train_full)
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签，正在修复...")
                y_train_full = y_train_full - 1
                y_test_full = y_test_full - 1

        # --- 5. 强制类型转换 ---
        # TabPFN 预处理需要类别列为字符串
        if cat_cols:
            for col in cat_cols:
                X_train_full[col] = X_train_full[col].astype(str)
                X_test_full[col] = X_test_full[col].astype(str)

        # 确保数值列是 float
        if num_cols:
            for col in num_cols:
                X_train_full[col] = pd.to_numeric(X_train_full[col], errors='coerce').fillna(0)
                X_test_full[col] = pd.to_numeric(X_test_full[col], errors='coerce').fillna(0)

        return X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def build_preprocess(num_cols, cat_cols):
    """
    构建预处理器
    """
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        # sparse_output=False 对 TabPFN 至关重要
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    
    return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)

def stratified_subsample_indices(y, sample_size, seed):
    """(保持不变)"""
    sample_size = int(sample_size)
    if len(y) <= sample_size:
        return np.arange(len(y))
    
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or (counts < 2).any():
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx_all = np.arange(len(y))
    try:
        for sub_idx, _ in sss.split(idx_all, y):
            return sub_idx
    except ValueError:
        np.random.seed(seed)
        return np.random.choice(idx_all, sample_size, replace=False)

def evaluate_metrics(y_true, y_pred, y_proba=None):
    """(保持不变)"""
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    if y_proba is not None:
        try:
            if y_proba.shape[1] == 2:
                res["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                res["auc_macro_ovr"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except:
            pass
    return res

# =========================
# 主流程
# =========================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seeds = [2022, 2023, 2024]
    results_all = []

    for seed in seeds:
        print(f"\n🚀 正在运行 seed = {seed}")
        cfg.seed = seed

        # --- 路径解析 (可选，保留原来的逻辑) ---
        data_root = cfg.get('data_base')
        if data_root:
            path_keys = [
                'labels_train_eval_tabular', 'labels_test_eval_tabular',
                'data_train_eval_tabular', 'data_test_eval_tabular',
                'field_lengths_tabular' # 确保这个也在更新列表里
            ]
            for key in path_keys:
                if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
                    cfg[key] = os.path.join(data_root, cfg[key])

        TRAIN_SAMPLE_THRESHOLD = cfg.get('train_sample_max', 8000)
        TEST_SAMPLE_THRESHOLD = cfg.get('test_sample_max', 2000)

        # 1. 加载数据 (使用修改后的函数)
        X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols = load_data(cfg)

        # 2. 预处理
        preprocess = build_preprocess(num_cols, cat_cols)

        # 3. 训练集采样
        if len(y_train_full) > TRAIN_SAMPLE_THRESHOLD:
            sample_size = TRAIN_SAMPLE_THRESHOLD
        else:
            sample_size = len(y_train_full)

        sub_idx = stratified_subsample_indices(y_train_full, sample_size, seed)
        X_train_sampled = X_train_full.iloc[sub_idx]
        y_train_sampled = y_train_full[sub_idx]

        # 4. 特征转换
        print("正在进行特征预处理...")
        X_train_np = preprocess.fit_transform(X_train_sampled)
        X_test_np  = preprocess.transform(X_test_full)
        print(f"特征矩阵形状: 训练集 {X_train_np.shape}, 测试集 {X_test_np.shape}")

        # 5. 模型初始化
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cfg.num_classes > 10:
            base_clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
            clf = ManyClassClassifier(estimator=base_clf, alphabet_size=10, random_state=seed)
        else:
            clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)

        # 6. 训练
        clf.fit(X_train_np, y_train_sampled)

        # 7. 测试集采样与评估
        X_test_eval = X_test_np
        y_test_eval = y_test_full

        if len(X_test_np) > TEST_SAMPLE_THRESHOLD:
            X_test_eval, _, y_test_eval, _ = train_test_split(
                X_test_np, y_test_full,
                train_size=TEST_SAMPLE_THRESHOLD,
                stratify=y_test_full,
                random_state=seed
            )
        
        test_proba = clf.predict_proba(X_test_eval)
        test_pred  = np.argmax(test_proba, axis=1)
        
        metrics = evaluate_metrics(y_test_eval, test_pred, test_proba)
        print(f"[RESULT] seed={seed} 指标: {json.dumps(metrics, indent=2)}")
        
        results_all.append({"seed": seed, "results": metrics})

    # 保存结果
    output_file = "/home/debian/TIP/mymodel/result/tabpfn_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"结果已保存: {output_file}")

if __name__ == "__main__":
    main()