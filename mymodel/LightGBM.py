import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
import sys
import torch
import os

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf

def load_and_preprocess_data(cfg: DictConfig):
    task = cfg.get('task', 'classification')
    print(f"--- 1. 正在为 Target: '{cfg.target}' (任务: {task}) (通用LGBM加载器) 加载数据 ---")

    try:
        # --- 1. 加载数据 ---
        X_train = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        X_test = pd.read_csv(cfg.data_val_eval_tabular, header=None)
        
        # 加载标签
        y_train = torch.load(cfg.labels_train_eval_tabular).numpy()
        y_test = torch.load(cfg.labels_val_eval_tabular).numpy()

        print("    数据加载成功。")

        # --- 2. 加载字段长度 (用于自动识别类别特征) ---
        # [!] 关键修改：读取 field_lengths
        all_field_lengths = torch.load(cfg.field_lengths_tabular)
        if isinstance(all_field_lengths, torch.Tensor):
            all_field_lengths = all_field_lengths.tolist()

        # 简单的校验
        if X_train.shape[1] != len(all_field_lengths):
            print(f"🔴 错误：CSV 列数 ({X_train.shape[1]}) 与 field_lengths 长度 ({len(all_field_lengths)}) 不一致！")
            sys.exit(1)

        # --- 3. 标签处理 (1-indexed -> 0-indexed) ---
        if task == 'classification':
            label_min = np.min(y_train)
            label_max = np.max(y_train)
            if label_min == 1 and label_max == cfg.num_classes:
                print(f"    [!] 警告：检测到 1-indexed 标签，正在修正...")
                y_train = y_train - 1
                y_test = y_test - 1

        # --- 4. 转换分类特征 (核心修改) ---
        
        # 自动识别：长度 > 1 的是类别特征
        cat_indices = [i for i, length in enumerate(all_field_lengths) if length > 1]
        
        print(f"    自动检测到 {len(cat_indices)} 个类别特征 (根据 field_lengths > 1)。")

        if len(cat_indices) > 0:
            # 为了避免 pandas 的 SettingWithCopyWarning 或类型混淆，
            # 建议给列重命名为字符串，这样处理起来更清晰
            X_train.columns = [str(i) for i in range(X_train.shape[1])]
            X_test.columns  = [str(i) for i in range(X_test.shape[1])]

            # 仅将检测到的类别列转换为 'category' 类型
            for idx in cat_indices:
                col_name = str(idx)
                # 转换为 category
                X_train[col_name] = X_train[col_name].astype('category')
                
                # 对齐测试集 (处理未知类别)
                # set_categories 确保测试集即使有未见过的类别也不会报错(会变成NaN)，
                # 或者确保其类别列表与训练集一致
                X_test[col_name] = pd.Categorical(X_test[col_name], categories=X_train[col_name].cat.categories, ordered=False)
            
            print("    已将类别特征转换为 pandas 'category' dtype。LightGBM 将自动识别它们。")
        else:
            print("    未检测到类别特征，所有列将作为数值处理。")

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        import traceback
        traceback.print_traceback()
        sys.exit(1)

    # --- 5. 确定问题类型 (保持不变) ---
    print("-" * 30)
    if task == 'classification':
        num_classes = cfg.get('num_classes', len(np.unique(y_train))) 
        if num_classes == 2:
            problem_type = 'binary'; objective = 'binary'; num_class_param = {}; scoring_metric = 'roc_auc'
        else:
            problem_type = 'multiclass'; objective = 'multiclass'; num_class_param = {'num_class': num_classes}; scoring_metric = 'accuracy'
    elif task == 'regression':
        problem_type = 'regression'; objective = 'regression_l2'; num_class_param = {}; scoring_metric = 'neg_root_mean_squared_error'
    else:
        print(f"错误: 不支持的任务类型 '{task}'"); sys.exit(1)

    print(f"LGBM Objective: {objective}, Scoring: {scoring_metric}")
    
    return X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric


def get_model_and_grid(problem_type, objective, num_class_param, seed):
    """
    根据问题类型获取LGBM模型和参数网格。
    """
    if problem_type in ['binary', 'multiclass']:
        model = lgb.LGBMClassifier(
            objective=objective,
            **num_class_param,
            random_state=seed,
            n_jobs=1,
            
            # --- ↓↓↓ 关键修改：添加下面一行 ↓↓↓ ---
            bagging_freq=1 # 只需要在这里激活 bagging
        )
    elif problem_type == 'regression':
        model = lgb.LGBMRegressor(
            objective=objective,
            random_state=seed,
            n_jobs=1,
            
            # --- ↓↓↓ 关键修改：添加下面一行 ↓↓↓ ---
            bagging_freq=1 # 只需要在这里激活 bagging
        )
    
    # --- ↓↓↓ 关键修改：修改 param_grid ↓↓↓ ---
    param_grid = {
        'num_leaves': [31, 127],
        'learning_rate': [0.01, 0.1],
        'min_child_samples': [20, 50, 100],
        'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1],
        
        # --- 将采样参数添加到网格搜索中 ---
        'feature_fraction': [0.8, 0.9], # 搜索 80% 或 90% 的特征
        'bagging_fraction': [0.8, 0.9]  # 搜索 80% 或 90% 的数据
    }
    
    return model, param_grid


def run_experiment(X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric, seed):
    """
    使用给定的随机种子运行一次模型训练和评估。
    """
    print(f"\n{'='*25} ---------------- 随机种子: {seed} ---------------- {'='*25}")
    
    model, param_grid = get_model_and_grid(problem_type, objective, num_class_param, seed)

    print(f"开始进行网格搜索 (评分指标: {scoring_metric})...")
    
    if problem_type == 'regression':
        cv_splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring_metric,
        cv=cv_splitter,
        n_jobs=-1, # GridSearchCV 使用所有核心
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print("网格搜索完成！")
    print(f"找到的最佳超参数: {grid_search.best_params_}")
    print(f"在交叉验证中的最佳 {scoring_metric}: {grid_search.best_score_:.4f}")
    print("-" * 30)

    print("使用最佳模型在测试集(验证集)上进行最终评估...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if problem_type in ['binary', 'multiclass']:
        y_pred_proba = best_model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        if problem_type == 'binary':
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
    
    elif problem_type == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        result_line = f"rmse:{rmse:.4f}"

    print("评估结果:")
    print(result_line)
    
    return result_line, grid_search.best_params_

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    # [!] 新增模块：解析数据路径 (与 PyTorch 脚本相同)
    # -----------------------------------------------------------------
    print("--- 1.A. 正在解析数据路径 ---")
    data_root = cfg.get('data_base') 
    
    if data_root is not None:
        print(f"    检测到 'data_root'，将为所有数据文件添加前缀: {data_root}")
        
        # 定义在 .yaml 中所有“需要”添加前缀的路径键
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
        
        # 遍历这些键，如果它们存在于 cfg 中，则更新路径
        for key in path_keys:
            if key in cfg and cfg[key] is not None:
                original_path = cfg[key]
                absolute_path = os.path.join(data_root, original_path)
                cfg[key] = absolute_path # [!] 直接修改 config 对象
            
    else:
        print("    未提供 'data_root'。将假定 config 中的路径已经是正确的 (绝对路径或相对于CWD)。")
    # -----------------------------------------------------------------


    print("\n--- 最终配置 (路径已解析): ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Hydra 工作目录: {os.getcwd()}")
    print("--------------------")

    # 1. 加载和预处理数据 (现在 cfg 包含绝对路径)
    X_train, y_train, X_test, y_test, problem_type, objective, num_class_param, scoring_metric = load_and_preprocess_data(cfg)

    # 允许 config.yaml 或命令行覆盖 'output_file_name'
    output_filename = '/root/mytip/mymodel/result/lgb_results.txt'
    seeds = [2022, 2023, 2024]
    
    # 3. 打开文件准备写入结果
    print(f"\n准备将结果写入到文件: {output_filename}")
    with open(output_filename, 'a') as f:
        f.write("--- 最终配置 ---\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("-" * 30 + "\n\n")

        # 4. 循环遍历所有随机种子
        for seed in seeds:
            result_line, best_params = run_experiment(
                X_train, y_train, X_test, y_test,
                problem_type, objective, num_class_param, scoring_metric,
                seed
            )
            
            # 5. 写入结果
            print(f"正在将种子 {seed} 的结果写入到 {output_filename}...")
            f.write(f"seed:{seed}\n")
            f.write(f"best_params: {best_params}\n")
            f.write(result_line + "\n\n")

    print(f"\n所有任务完成！结果已全部保存在 '{os.getcwd()}/{output_filename}' 中。")


if __name__ == "__main__":
    main()