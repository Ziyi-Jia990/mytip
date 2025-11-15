# eval_tabpfn.py
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
import torch # 新增

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# <--- 修改导入：增加了 roc_auc_score ---
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from tabpfn import TabPFNClassifier
from tabpfn_extensions.many_class.many_class_classifier import ManyClassClassifier

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf

# =========================
# 工具函数（修改为从 cfg 读取）
# =========================

# 从 cfg 中读取固定的阈值，如果cfg中没有，则使用默认值
TEXT_LENGTH_DROP_THRESHOLD = 30
HIGH_CARDINALITY_THRESHOLD = 200
N_ENSEMBLE_CONFIGURATIONS = 16


def load_data(cfg: DictConfig):
    """
    (重构版) 
    通用数据加载器，用于加载已预处理的 .csv 和 .pt 文件。
    """
    target = cfg.target
    print(f"[INFO] 正在加载 target: {target} (通用加载器)")

    try:
        # --- 1. 加载数据 ---
        # 训练集 = train_eval_tabular
        # 测试集 = val_eval_tabular (与 LGBM 脚本保持一致，使用验证集进行评估)
        
        # 加载训练集
        X_train_full = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        # [!] 移除 'weights_only' 以兼容旧版 PyTorch
        y_train_tensor = torch.load(cfg.labels_train_eval_tabular) 
        y_train_full = y_train_tensor.numpy()

        # 加载测试集 (我们使用 'val' 数据集)
        X_test_full = pd.read_csv(cfg.data_test_eval_tabular, header=None)
        y_test_tensor = torch.load(cfg.labels_test_eval_tabular)
        y_test_full = y_test_tensor.numpy()

        # --- 2. 检查并修复 1-indexed 标签 ---
        if cfg.task == 'classification':
            label_min = np.min(y_train_full)
            label_max = np.max(y_train_full)
            
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签 (min={label_min}, max={label_max})。")
                print("        正在减去 1 使其变为 0-indexed。")
                y_train_full = y_train_full - 1
                y_test_full = y_test_full - 1
            elif label_min < 0 or label_max >= cfg.num_classes:
                print(f"🔴 错误：标签越界！")
                print(f"       模型有 {cfg.num_classes} 个类别 (预期 0 到 {cfg.num_classes - 1})")
                print(f"       但标签中发现 最小值={label_min}, 最大值={label_max}")
                sys.exit(1)
        
        # --- 3. 定义列名和类型 ---
        num_con = cfg.num_con
        num_cat = cfg.num_cat
        
        if X_train_full.shape[1] != (num_con + num_cat):
            print(f"🔴 错误：加载的 X_train 有 {X_train_full.shape[1]} 列, 但 config 预期 {num_con + num_cat} 列。")
            sys.exit(1)

        # 创建列名
        num_cols = [f"num_{i}" for i in range(num_con)]
        cat_cols = [f"cat_{i}" for i in range(num_cat)]
        all_cols = num_cols + cat_cols

        X_train_full.columns = all_cols
        X_test_full.columns = all_cols
        
        # --- 4. 关键：强制转换类型 ---
        # 我们必须强制 cat_cols 为 'object'/'str'，
        # 这样 build_preprocess 中的 `is_numeric_dtype` 才能正确工作。
        for col in cat_cols:
            X_train_full[col] = X_train_full[col].astype(str)
            X_test_full[col] = X_test_full[col].astype(str)

        # 列对齐 (在 build_preprocess 之前是多余的，但保留以防万一)
        X_test_full = X_test_full[X_train_full.columns]

        print(f"[INFO] 数值列 (基于config): {num_cols}")
        print(f"[INFO] 分类列 (基于config): {cat_cols}")

        return X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols

    except FileNotFoundError as e:
        print(f"🔴 错误：找不到文件 {e.filename}。")
        print("    请确保 config 中的路径正确，并且预处理脚本已成功运行。")
        sys.exit(1)
    except KeyError as e:
        print(f"🔴 错误：Config 文件中缺少关键的键: {e}")
        print("    请确保 cfg 包含 'data_train_eval_tabular', 'labels_train_eval_tabular', 'data_val_eval_tabular', 'labels_val_eval_tabular', 'num_con', 'num_cat', 'task', 'num_classes'")
        sys.exit(1)
    except Exception as e:
        print(f"🔴 加载数据时发生意外错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def build_preprocess(num_cols, cat_cols):
    """
    修改：确保输出为 TabPFN 需要的密集 (dense) 矩阵。
    """
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        # 关键修复：sparse=True -> sparse_output=False
        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)) 
    
    # 关键修复：sparse_threshold=1.0 -> 0.0
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)

def stratified_subsample_indices(y, sample_size, seed):
    """
    从 y 中获取用于采样的索引 (您的代码已正确)
    """
    # 确保 sample_size 是整数
    sample_size = int(sample_size)
    if len(y) <= sample_size:
        return np.arange(len(y))
    
    # 确保 y 中至少有2个类别，或者足够的样本
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2 or (counts < 2).any():
        print("[WARNING] 类别太少或样本不足，无法进行分层采样，退回到随机采样。")
        np.random.seed(seed)
        return np.random.choice(np.arange(len(y)), sample_size, replace=False)

    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    idx_all = np.arange(len(y))
    try:
        for sub_idx, _ in sss.split(idx_all, y):
            return sub_idx
    except ValueError as e:
        print(f"[WARNING] 分层采样失败 ({e})，退回到随机采样。")
        np.random.seed(seed)
        return np.random.choice(idx_all, sample_size, replace=False)

# <--- 函数已修改 (增加AUC) ---
def evaluate_metrics(y_true, y_pred, y_proba=None):
    """
    (已修改：增加了 AUC 计算)
    """
    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_proba is not None:
        # 获取概率矩阵中的类别数 和 真实标签中的类别数
        n_classes_proba = y_proba.shape[1]
        unique_true_classes = np.unique(y_true)
        n_classes_true = len(unique_true_classes)

        # --- 1. LogLoss (原有逻辑) ---
        try:
            # 确保 y_proba 的列数与类别数一致
            # 确保 y_true 中的标签在 [0, n_classes-1] 范围内
            if y_true.max() >= n_classes_proba:
                print(f"[WARNING] y_true 包含标签 {y_true.max()}，但 y_proba 只有 {n_classes_proba} 列。LogLoss 可能不准确。")
            
            res["log_loss"] = log_loss(y_true, y_proba, labels=np.arange(n_classes_proba))
        except Exception as e:
            print(f"[WARNING] 无法计算 LogLoss: {e}")
            pass

        # --- 2. AUC (新增逻辑) ---
        try:
            # 检查 y_true 中是否只有一个类别，这会导致 AUC 无法计算
            if n_classes_true < 2:
                print(f"[WARNING] y_true 中只有一个类别 ({unique_true_classes})，跳过 AUC 计算。")
            
            # 情况 A: 二分类 (y_proba 有 2 列)
            elif n_classes_proba == 2:
                # roc_auc_score 需要 y_true 和 *正类*的概率
                res["auc"] = roc_auc_score(y_true, y_proba[:, 1])
            
            # 情况 B: 多分类 (y_proba 有 >2 列)
            else:
                res["auc_macro_ovr"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                res["auc_weighted_ovr"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                # 您也可以选择 'ovo'
                # res["auc_macro_ovo"] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')

        except Exception as e:
            print(f"[WARNING] 无法计算 AUC: {e}")
            pass

    return res
# <--- 函数修改结束 ---

# =========================
# 主流程 (Hydra)
# =========================

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    seeds = [2022, 2023, 2024]
    results_all = []

    for seed in seeds:
        print(f"\n==============================")
        print(f"🚀 正在运行 seed = {seed}")
        print(f"==============================")

        cfg.seed = seed   # 动态修改配置中的 seed

        # -------------------------------------------------
        # 下面是你原来 main(cfg) 内部的整个流程——保持不变
        # -------------------------------------------------

        print("--- 1.A. 正在解析数据路径 ---")
        data_root = cfg.get('data_base') 
        if data_root is not None:
            print(f"    检测到 'data_root'，将为所有数据文件添加前缀: {data_root}")
            path_keys = [
                'labels_train', 'labels_val',
                'data_train_imaging', 'data_val_imaging',
                'data_train_tabular', 'data_val_tabular',
                'field_lengths_tabular',
                'data_train_eval_tabular', 'labels_train_eval_tabular',
                'data_val_eval_tabular', 'labels_val_eval_tabular',
                'data_test_eval_tabular', 'labels_test_eval_tabular',
                'data_train_eval_imaging', 'labels_train_eval_imaging',
                'data_val_eval_imaging', 'labels_val_eval_imaging',
                'data_test_eval_imaging', 'labels_test_eval_imaging'
            ]
            for key in path_keys:
                if key in cfg and cfg[key] is not None:
                    cfg[key] = os.path.join(data_root, cfg[key])
        else:
            print("    未提供 'data_root'。将假定 config 中的路径已经是正确的。")

        print("\n--- 最终配置 (路径已解析): ---")
        print(OmegaConf.to_yaml(cfg))
        print("--------------------")
        print(f"Hydra 工作目录: {os.getcwd()}")
        print("--------------------")

        TRAIN_SAMPLE_THRESHOLD = cfg.get('train_sample_max', 8000)
        TEST_SAMPLE_THRESHOLD = cfg.get('test_sample_max', 2000)

        # 读取数据
        X_train_full, y_train_full, X_test_full, y_test_full, num_cols, cat_cols = load_data(cfg)

        # 预处理
        preprocess = build_preprocess(num_cols, cat_cols)

        # 采样
        if len(y_train_full) > TRAIN_SAMPLE_THRESHOLD:
            sample_size = TRAIN_SAMPLE_THRESHOLD
        else:
            sample_size = len(y_train_full)

        sub_idx = stratified_subsample_indices(y_train_full, sample_size, seed)
        X_train_sampled = X_train_full.iloc[sub_idx]
        y_train_sampled = y_train_full[sub_idx]

        X_train_np = preprocess.fit_transform(X_train_sampled)
        X_test_np  = preprocess.transform(X_test_full)

        n_ensemble = N_ENSEMBLE_CONFIGURATIONS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"[INFO] 使用设备: {device}")

        # ---- 关键：TabPFN 不要 batch_size，避免报错 ----
        if cfg.num_classes > 10:
            print(f"[INFO] 多分类，使用 ManyClassClassifier")
            base_clf = TabPFNClassifier(
                n_estimators=n_ensemble,
                device=device
            )
            clf = ManyClassClassifier(
                estimator=base_clf,
                alphabet_size=10,
                random_state=seed,
                verbose=1,
            )
        else:
            clf = TabPFNClassifier(
                n_estimators=n_ensemble,
                device=device
            )

        print("Fitting TabPFN...")
        clf.fit(X_train_np, y_train_sampled)
        print("Fit complete.")

        # 测试集采样
        X_test_sampled = X_test_np
        y_test_sampled = y_test_full

        if len(X_test_np) > TEST_SAMPLE_THRESHOLD:
            try:
                X_test_sampled, _, y_test_sampled, _ = train_test_split(
                    X_test_np, y_test_full,
                    train_size=TEST_SAMPLE_THRESHOLD,
                    stratify=y_test_full,
                    random_state=seed
                )
            except Exception as e:
                print("测试集采样失败，使用完整测试集。")

        # 预测
        test_proba = clf.predict_proba(X_test_sampled)
        test_pred  = np.argmax(test_proba, axis=1)

        metrics = evaluate_metrics(y_test_sampled, test_pred, test_proba)

        print(f"[RESULT] seed={seed} 测试集指标：")
        print(json.dumps(metrics, indent=2))

        # 保存当前 seed 的结果到列表
        results_all.append({
            "seed": seed,
            "results": metrics
        })

    # ---------------------------------------
    # 统一写入结果文件（一次性写合法 JSON）
    # ---------------------------------------
    output_file = "/home/debian/TIP/mymodel/result/tabpfn_results.json"
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f"\n🎉 所有 seed 运行完成，结果已写入：{output_file}\n")



if __name__ == "__main__":
    main()
