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
import random

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# [修改] 引入回归指标
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# [修改] 引入 Regressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
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
        
        # [修改] 回归任务保持 float，分类任务转 numpy
        if cfg.task == 'regression':
            y_train_full = y_train_tensor.float().numpy()
        else:
            y_train_full = y_train_tensor.numpy()

        X_test_full = pd.read_csv(cfg.data_test_eval_tabular, header=None)
        y_test_tensor = torch.load(cfg.labels_test_eval_tabular)
        
        if cfg.task == 'regression':
            y_test_full = y_test_tensor.float().numpy()
        else:
            y_test_full = y_test_tensor.numpy()

        # --- 2. 加载 field_lengths 并计算索引 ---
        field_lengths_path = cfg.field_lengths_tabular
        print(f"[INFO] 读取字段长度文件: {field_lengths_path}")
        
        try:
            field_lengths = torch.load(field_lengths_path)
            if isinstance(field_lengths, torch.Tensor):
                field_lengths = field_lengths.cpu().numpy()
        except Exception:
            field_lengths = np.load(field_lengths_path)
        
        field_lengths = np.array(field_lengths).flatten()
        
        n_cols_data = X_train_full.shape[1]
        n_cols_lengths = len(field_lengths)
        if n_cols_data != n_cols_lengths:
            print(f"🔴 错误：CSV 列数 ({n_cols_data}) 与 field_lengths 长度 ({n_cols_lengths}) 不匹配！")
            sys.exit(1)

        con_indices = [i for i, fl in enumerate(field_lengths) if fl == 1]
        cat_indices = [i for i, fl in enumerate(field_lengths) if fl > 1]
        
        print(f"[INFO] 自动检测结果:")
        print(f"      - 数值列数量: {len(con_indices)}")
        print(f"      - 类别列数量: {len(cat_indices)}")

        # --- 3. 定义列名 ---
        all_col_names = [f"col_{i}" for i in range(n_cols_data)]
        X_train_full.columns = all_col_names
        X_test_full.columns = all_col_names

        num_cols = [all_col_names[i] for i in con_indices]
        cat_cols = [all_col_names[i] for i in cat_indices]

        # --- 4. 标签处理 (1-indexed -> 0-indexed) ---
        # [修改] 只有分类任务才执行此操作
        if cfg.task == 'classification':
            label_min = np.min(y_train_full)
            label_max = np.max(y_train_full)
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签，正在修复...")
                y_train_full = y_train_full - 1
                y_test_full = y_test_full - 1

        # --- 5. 强制类型转换 ---
        if cat_cols:
            for col in cat_cols:
                X_train_full[col] = X_train_full[col].astype(str)
                X_test_full[col] = X_test_full[col].astype(str)

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
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))
    
    return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)

def get_subsample_indices(y, sample_size, seed, task):
    """
    [修改] 通用采样函数：
    - 分类任务：分层采样
    - 回归任务：随机采样
    """
    sample_size = int(sample_size)
    if len(y) <= sample_size:
        return np.arange(len(y))
    
    # 1. 回归任务直接随机采样
    if task == 'regression':
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    # 2. 分类任务逻辑
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

def evaluate_metrics(y_true, y_pred, task, y_proba=None):
    """
    [修改] 支持回归和分类指标
    """
    res = {}
    
    if task == 'classification':
        res["accuracy"] = accuracy_score(y_true, y_pred)
        res["macro_f1"] = f1_score(y_true, y_pred, average='macro')
        res["weighted_f1"] = f1_score(y_true, y_pred, average='weighted')
        
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    res["auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    res["auc_macro_ovr"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            except:
                pass
                
    elif task == 'regression':
        # [新增] 回归指标
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        res["rmse"] = rmse
        res["mae"] = mae
        res["r2"] = r2
        
    return res

# =========================
# 主流程
# =========================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seeds = [2022, 2023, 2024]
    results_all = []

    # 确保 cfg 中有 task 字段
    if 'task' not in cfg:
        print("⚠️ Config 中缺少 'task' 字段，默认设为 'classification'")
        cfg.task = 'classification'

    for seed in seeds:
        print(f"\n🚀 正在运行 seed = {seed} | Task: {cfg.task}")
        cfg.seed = seed

        # --- 路径解析 ---
        data_root = cfg.get('data_base')
        if data_root:
            path_keys = [
                'labels_train_eval_tabular', 'labels_test_eval_tabular',
                'data_train_eval_tabular', 'data_test_eval_tabular',
                'field_lengths_tabular'
            ]
            for key in path_keys:
                if key in cfg and cfg[key] and not os.path.isabs(cfg[key]):
                    cfg[key] = os.path.join(data_root, cfg[key])

        TRAIN_SAMPLE_THRESHOLD = cfg.get('train_sample_max', 8000)
        TEST_SAMPLE_THRESHOLD = cfg.get('test_sample_max', 2000)

        # 1. 加载数据
        X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols = load_data(cfg)

        # 2. 预处理
        preprocess = build_preprocess(num_cols, cat_cols)

        # 3. 训练集采样 (自动根据 task 选择采样方式)
        if len(y_train_full) > TRAIN_SAMPLE_THRESHOLD:
            sample_size = TRAIN_SAMPLE_THRESHOLD
        else:
            sample_size = len(y_train_full)

        sub_idx = get_subsample_indices(y_train_full, sample_size, seed, cfg.task)
        X_train_sampled = X_train_full.iloc[sub_idx]
        y_train_sampled = y_train_full[sub_idx]

        # 4. 特征转换
        print("正在进行特征预处理...")
        X_train_np = preprocess.fit_transform(X_train_sampled)
        X_test_np  = preprocess.transform(X_test_full)
        print(f"特征矩阵形状: 训练集 {X_train_np.shape}, 测试集 {X_test_np.shape}")

        # 5. 模型初始化 [修改：根据任务选择模型]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if cfg.task == 'classification':
            if cfg.num_classes > 10:
                base_clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
                clf = ManyClassClassifier(estimator=base_clf, alphabet_size=10, random_state=seed)
            else:
                clf = TabPFNClassifier(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
        elif cfg.task == 'regression':
            # [新增] 回归模型
            print("正在初始化 TabPFNRegressor...")
            clf = TabPFNRegressor(n_estimators=N_ENSEMBLE_CONFIGURATIONS, device=device)
        else:
            raise ValueError(f"未知的任务类型: {cfg.task}")

        # 6. 训练
        clf.fit(X_train_np, y_train_sampled)

        # 7. 测试集采样与评估
        X_test_eval = X_test_np
        y_test_eval = y_test_full

        if len(X_test_np) > TEST_SAMPLE_THRESHOLD:
            # 回归用普通 split，分类用 stratify
            if cfg.task == 'classification':
                stratify_target = y_test_full
            else:
                stratify_target = None # 回归不能 stratify

            X_test_eval, _, y_test_eval, _ = train_test_split(
                X_test_np, y_test_full,
                train_size=TEST_SAMPLE_THRESHOLD,
                stratify=stratify_target,
                random_state=seed
            )
        
        # 8. 预测 [修改：区分分类和回归]
        test_proba = None
        test_pred = None

        if cfg.task == 'classification':
            test_proba = clf.predict_proba(X_test_eval)
            test_pred  = np.argmax(test_proba, axis=1)
        else:
            # 回归没有 predict_proba
            test_pred = clf.predict(X_test_eval)
        
        # 9. 计算指标
        metrics = evaluate_metrics(y_test_eval, test_pred, cfg.task, test_proba)
        print(f"[RESULT] seed={seed} 指标: {json.dumps(metrics, indent=2)}")
        
        results_all.append({"seed": seed, "results": metrics})

    # 保存结果
    output_file = "result/tabpfn_results.json" # 建议用相对路径或从 cfg 读取
    if os.path.exists("/home/debian/TIP/mymodel/result/"): # 如果原始绝对路径存在
        output_file = "/home/debian/TIP/mymodel/result/tabpfn_results.json"
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)
    print(f"结果已保存: {output_file}")

if __name__ == "__main__":
    main() 