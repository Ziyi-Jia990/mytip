import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import rtdl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
import time
import os
import random
import sys
import json

# --- Hydra 导入 ---
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path # 

# --- 0. 配置与设置随机种子 ---
def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 自定义模型类 ---
class MyFTTransformer(nn.Module):
    def __init__(self, ft_transformer_module):
        super().__init__()
        self.ft_transformer = ft_transformer_module

    def forward(self, x_num, x_cat):
        if x_num is not None and x_num.shape[1] == 0:
            x_num = None
            
        out = self.ft_transformer(x_num, x_cat)

        if isinstance(out, tuple):
            x_embed, x_out = out   
        else:
            x_out = out           

        # 回归任务 d_out=1 时，做个 squeeze
        if x_out.shape[-1] == 1:
            return x_out.squeeze(-1)
        return x_out


# --- 1. 数据加载与预处理 (重构为函数) ---
def load_and_preprocess_data(cfg: DictConfig, batch_size: int):
    print(f"--- 1. 正在为数据集: '{cfg.target}' (通用加载器) 加载数据 ---")

    try:
        # --- 1. 加载字段长度 (基数) ---
        # 你的逻辑： == 1 是连续特征， > 1 是类别特征
        all_field_lengths = torch.load(cfg.field_lengths_tabular)
        
        # 转换为 list 或 numpy 方便处理
        if isinstance(all_field_lengths, torch.Tensor):
            all_field_lengths = all_field_lengths.cpu().tolist()
            
        # === 核心修改：动态识别索引 ===
        con_indices = [i for i, length in enumerate(all_field_lengths) if length == 1]
        cat_indices = [i for i, length in enumerate(all_field_lengths) if length > 1]
        
        num_con = len(con_indices)
        num_cat = len(cat_indices)
        
        # 提取分类特征的基数 (Cardinalities)，顺序必须与 cat_indices 对应
        cat_cardinalities = [all_field_lengths[i] for i in cat_indices]

        print(f"    自动检测结果: {num_con} 个连续特征, {num_cat} 个分类特征。")

        # --- 2. 加载特征 (CSV) ---
        train_df = pd.read_csv(cfg.data_train_eval_tabular, header=None)
        val_df = pd.read_csv(cfg.data_val_eval_tabular, header=None)
        test_df = pd.read_csv(cfg.data_test_eval_tabular, header=None)

        # 简单的校验
        if train_df.shape[1] != len(all_field_lengths):
            print(f"🔴 错误：CSV 列数 ({train_df.shape[1]}) 与 field_lengths 长度 ({len(all_field_lengths)}) 不一致！")
            sys.exit(1)

        # --- 3. 加载标签 ---
        y_train = torch.load(cfg.labels_train_eval_tabular)
        y_val = torch.load(cfg.labels_val_eval_tabular)
        y_test = torch.load(cfg.labels_test_eval_tabular)

    except Exception as e:
        print(f"🔴 加载数据时发生错误: {e}")
        sys.exit(1)

    # --- 4. 拆分特征并转换为 Tensors (修改版) ---
    
    def split_and_convert_to_tensors(df, y_tensor):
        # === 使用索引提取数据 ===
        # 即使 con_indices 为空，df.iloc[:, []] 也会返回空 DF，不会报错
        X_num_df = df.iloc[:, con_indices]
        X_cat_df = df.iloc[:, cat_indices]
        
        # === 关键修复：解决 rtdl 报错 ===
        # 如果没有数值特征，直接设为 None，而不是空 Tensor
        if len(con_indices) > 0:
            X_num_tensor = torch.tensor(X_num_df.values.astype(np.float32))
        else:
            X_num_tensor = None  # <--- 这里直接给 None，解决之前的 AssertionError

        # 类别特征处理
        X_cat_tensor = torch.tensor(X_cat_df.values.astype(np.int64))
        
        # 标签处理
        if cfg.task == 'classification':
            y_tensor = y_tensor.long()
        else:
            y_tensor = y_tensor.float()

        return X_num_tensor, X_cat_tensor, y_tensor

    print("    正在根据 field_lengths 拆分并转换数据...")
    # 这里原来的 Dataset 可能会报错，因为 TensorDataset 不支持 None
    # 我们需要下面特别处理 TensorDataset
    
    X_train_num, X_train_cat, y_train = split_and_convert_to_tensors(train_df, y_train)
    X_val_num, X_val_cat, y_val = split_and_convert_to_tensors(val_df, y_val)
    X_test_num, X_test_cat, y_test = split_and_convert_to_tensors(test_df, y_test)
    
    # ==================================================================
    # == 创建 DataLoaders (需要处理 None 的情况)
    # ==================================================================
    
    # 定义一个安全的 Dataset 类，允许 x_num 为 None
    class SafeTabularDataset(torch.utils.data.Dataset):
        def __init__(self, x_num, x_cat, y):
            self.x_num = x_num
            self.x_cat = x_cat
            self.y = y
            
        def __len__(self):
            return len(self.y)
            
        def __getitem__(self, idx):
            # 如果 x_num 是 None，返回一个占位符或者在 collate_fn 里处理
            # 简单起见，这里我们返回一个空的 tensor (如果是None)，
            # 但既然我们上面为了解决 rtdl 改成了 None，这里为了 DataLoader 方便，
            # 我们可以保留 None，但在取出时要注意。
            
            # 更加简便的方法：
            # 如果 x_num 是 None，我们就不把它放进 TensorDataset，而是造一个自定义 Dataset
            num_val = self.x_num[idx] if self.x_num is not None else torch.empty(0)
            return num_val, self.x_cat[idx], self.y[idx]

    # 为了不引入复杂的 Dataset 类，最简单的 Hack 方法：
    # 如果 x_num 是 None，我们还是变回空 Tensor 存入 DataLoader，
    # 但是在 Model 的 forward 里进行判断（就像上一条回复建议的那样）。
    
    # 修正策略：
    # 1. 这里 DataLoader 里还是存空 Tensor (为了方便批处理)
    # 2. Model 里加判断 (上一条回复的方案)
    # 这样改动最小。
    
    # 重新修正 split_and_convert_to_tensors 的返回值，改回返回 Tensor
    if X_train_num is None: X_train_num = torch.empty((len(y_train), 0))
    if X_val_num is None: X_val_num = torch.empty((len(y_val), 0))
    if X_test_num is None: X_test_num = torch.empty((len(y_test), 0))

    try:
        train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
        val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
        test_dataset = TensorDataset(X_test_num, X_test_cat, y_test)
    except Exception as e:
        print(f"🔴 创建 Dataset 出错: {e}")
        sys.exit(1)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 3. 准备模型输入 ---
    
    # 模型输出维度
    d_out = cfg.num_classes
    
    # 返回所有必要组件
    model_inputs = {
        "n_num_features": num_con,          # 数值特征的数量
        "cat_cardinalities": cat_cardinalities, # 分类特征基数列表
        "d_out": d_out,
        "task": cfg.task
    }
    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    return loaders, model_inputs

# --- 3. 定义模型创建函数 (已修正：补全 rtdl.Transformer 所需的参数) ---
def create_model(params, n_num_features, cat_cardinalities, d_out, device):
    base_ft_transformer = rtdl.FTTransformer(
        feature_tokenizer=rtdl.modules.FeatureTokenizer(
            n_num_features=n_num_features, 
            cat_cardinalities=cat_cardinalities, 
            d_token=params['d_token']
        ),
        transformer=rtdl.modules.Transformer(
            d_token=params['d_token'],
            n_blocks=params['n_blocks'],
            attention_dropout=params['attention_dropout'],
            ffn_d_hidden=params['ffn_d_hidden'],
            ffn_dropout=params['ffn_dropout'],
            residual_dropout=params['residual_dropout'],

            attention_n_heads=8,
            attention_initialization='kaiming',
            attention_normalization='LayerNorm',
            ffn_activation='ReLU',
            ffn_normalization='LayerNorm',
            prenormalization=True,
            first_prenormalization=False,
            last_layer_query_idx=[-1],
            n_tokens=None,
            kv_compression_ratio=None,
            kv_compression_sharing=None,
            head_activation=nn.Identity,
            head_normalization=nn.Identity,

            # 🔴 关键：这里的 d_out 一定要是 “最终输出维度”，也就是 num_classes
            d_out=d_out,
        ),
    )

    model = MyFTTransformer(
        ft_transformer_module=base_ft_transformer,
    ).to(device)

    return model


# --- 4. 辅助函数：获取损失函数和评估指标 ---
def create_loss_fn(task, device):
    if task == 'classification':
        return nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        return nn.MSELoss().to(device)
    else:
        raise ValueError(f"未知的任务类型: {task}")

def get_scoring_info(task):
    """获取阶段一搜索所需的评估指标和优化方向"""
    if task == 'classification':
        return 'accuracy', 'max' # 指标名称, 优化方向
    elif task == 'regression':
        return 'rmse', 'min'
    else:
        raise ValueError(f"未知的任务类型: {task}")

# --- 5. 定义训练与评估函数 ---

# 阶段一函数：快速搜索
def search_for_best_params(param_combinations, cfg, seed, loaders, model_inputs, device):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段一：快速超参数搜索 (15 epochs) " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    scoring_metric, mode = get_scoring_info(task)
    best_score = -float('inf') if mode == 'max' else float('inf')
    best_params = None
    
    for i, params in enumerate(param_combinations):
        print(f"\n--- [试验 {i+1}/{len(param_combinations)}] ---")
        print(f"测试参数: {params}")
        
        set_seed(seed)
        model = create_model(params, n_num, cats, d_out, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        loss_fn = create_loss_fn(task, device)
        
        # 训练固定的15个epoch
        for epoch in range(15):
            model.train()
            for x_num_batch, x_cat_batch, y_batch in train_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(x_num_batch, x_cat_batch)
                # print("y_pred shape:", y_pred.shape, "dtype:", y_pred.dtype)
                # print("y_batch shape:", y_batch.shape, "dtype:", y_batch.dtype)

                loss = loss_fn(y_pred, y_batch.long())
                loss.backward()
                optimizer.step()
                

        # 在验证集上评估
        model.eval()
        val_preds_proba = []
        val_labels = []
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                
                # 统一处理: proba 用于分类, value 用于回归
                if task == 'classification':
                    val_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
                else:
                    val_preds_proba.append(y_pred.cpu().numpy()) # (N,) or (N, 1)
                val_labels.append(y_batch.cpu().numpy())
        
        val_preds_proba = np.concatenate(val_preds_proba)
        val_labels = np.concatenate(val_labels)
        
        # 动态计算得分
        current_score = 0.0
        if task == 'classification':
            val_preds_class = np.argmax(val_preds_proba, axis=1)
            current_score = accuracy_score(val_labels, val_preds_class)
        elif task == 'regression':
            current_score = np.sqrt(mean_squared_error(val_labels, val_preds_proba.squeeze()))
        
        print(f"试验 {i+1} 验证集 {scoring_metric}: {current_score:.4f}")
        
        if (mode == 'max' and current_score > best_score) or \
           (mode == 'min' and current_score < best_score):
            best_score = current_score
            best_params = params
            print(f"  (发现新的最佳参数!)")
            
    print("\n" + "-"*10 + " 阶段一搜索完成 " + "-"*10)
    print(f"最佳验证集 {scoring_metric}: {best_score:.4f}")
    print(f"选定的最佳参数: {best_params}")
    return best_params

# 阶段二函数：使用早停充分训练
def train_final_model(best_params, cfg, seed, loaders, model_inputs, device, 
                      patience: int, max_epochs: int):
    print("\n" + "-"*10 + f" [种子 {seed}] 阶段二：使用早停机制充分训练最佳模型 " + "-"*10)
    
    train_loader, val_loader = loaders['train'], loaders['val']
    n_num, cats, d_out, task = model_inputs.values()
    
    set_seed(seed)
    model = create_model(best_params, n_num, cats, d_out, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    loss_fn = create_loss_fn(task, device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'best_model_seed_{seed}.pt' 
    # [!] 使用来自 json 的参数
    # max_epochs = cfg.hyperparams.max_epochs (移除)
    # patience = cfg.hyperparams.patience (移除)

    for epoch in range(max_epochs):
        model.train()
        for x_num_batch, x_cat_batch, y_batch in train_loader:
            x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_num_batch, x_cat_batch)
            loss = loss_fn(y_pred, y_batch.long())
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_num_batch, x_cat_batch, y_batch in val_loader:
                x_num_batch, x_cat_batch, y_batch = x_num_batch.to(device), x_cat_batch.to(device), y_batch.to(device)
                y_pred = model(x_num_batch, x_cat_batch)
                val_loss += loss_fn(y_pred, y_batch.long()).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{max_epochs}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  (验证集损失连续 {patience} 个epoch未改善，触发早停!)")
                break
    
    print(f"加载在验证集上性能最佳的模型 (来自 {best_model_path})...")
    model.load_state_dict(torch.load(best_model_path))
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
        
    return model

# 阶段三函数：在测试集上评估
def evaluate_final_model(cfg, final_model, test_loader, task, device):
    final_model.eval()
    all_preds_proba, all_labels = [], []
    with torch.no_grad():
        for x_num_batch, x_cat_batch, y_batch in test_loader:
            x_num_batch, x_cat_batch = x_num_batch.to(device), x_cat_batch.to(device)
            
            y_pred = final_model(x_num_batch, x_cat_batch)
            
            if task == 'classification':
                all_preds_proba.append(y_pred.softmax(dim=1).cpu().numpy())
            else:
                all_preds_proba.append(y_pred.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            
    all_preds_proba = np.concatenate(all_preds_proba)
    all_labels = np.concatenate(all_labels)

    metrics_dict = {}
    result_line = ""

    if task == 'classification':
        all_preds_class = np.argmax(all_preds_proba, axis=1)
        acc = accuracy_score(all_labels, all_preds_class)
        # auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovr', average='macro')
        if cfg.num_classes == 2:
            auc = roc_auc_score(all_labels, all_preds_proba[:, 1])
        else:
            # 如果未来有多分类需求，保留原逻辑
            auc = roc_auc_score(all_labels, all_preds_proba, multi_class='ovr', average='macro')
        macro_f1 = f1_score(all_labels, all_preds_class, average='macro')
        
        metrics_dict = {'acc': acc, 'auc': auc, 'macro-F1': macro_f1}
        result_line = f"acc:{acc:.4f},auc:{auc:.4f},macro-F1:{macro_f1:.4f}"
        
    elif task == 'regression':
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds_proba.squeeze()))
        metrics_dict = {'rmse': rmse}
        result_line = f"rmse:{rmse:.4f}"

    print(f"最终测试集性能: {result_line}")
    return metrics_dict, result_line

# --- 6. 主执行流程 (Hydra Main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    print("--- 1.A. 正在解析数据路径 ---")
    
    # 1. 从命令行获取 'data_root'
    #    (如果未提供，data_root 将为 None)
    data_root = cfg.get('data_base')

    if data_root is not None:
        print(f"    检测到 'data_root'，将为所有数据文件添加前缀: {data_root}")
        
        # 2. 定义在 .yaml 中所有“需要”添加前缀的路径键
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
        
        # 3. 遍历这些键，如果它们存在于 cfg 中，则更新路径
        for key in path_keys:
            if key in cfg and cfg[key] is not None:
                original_path = cfg[key]
                absolute_path = os.path.join(data_root, original_path)
                cfg[key] = absolute_path # [!] 直接修改 config 对象
                # print(f"        {key}: {original_path} -> {absolute_path}") # (可选：用于调试)
            
    else:
        print("    未提供 'data_root'。将假定 config 中的路径已经是正确的 (绝对路径或相对于CWD)。")
    
    print("--- 最终配置: ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------")
    print(f"Hydra 工作目录: {os.getcwd()}")
    print("--------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'正在使用设备: {device}')
    
    original_cwd = get_original_cwd() 
    model_config_filename = 'ftt_model_config.json' # [!] 文件名修改
    model_config_path = os.path.join(original_cwd, model_config_filename)
    
    print(f"--- 正在从 {model_config_path} 加载模型配置 ---")
    try:
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        search_space = model_config['search_space']
        hyperparams = model_config['hyperparams']
        print("模型配置 (hyperparams 和 search_space) 加载成功。")
    except FileNotFoundError:
        print(f"错误: 找不到模型配置文件: {model_config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"错误: 模型配置文件 '{model_config_path}' 缺少键: {e}")
        sys.exit(1)
    
    
    # 2. 将 hyperparams 提取到变量
    seeds = list(hyperparams['seeds'])
    batch_size = hyperparams['batch_size']
    D_TOKEN = hyperparams['d_token']
    LEARNING_RATE = hyperparams['learning_rate']
    WEIGHT_DECAY = hyperparams['weight_decay']
    N_TRIALS = hyperparams['n_trials']
    patience = hyperparams['patience']
    max_epochs = hyperparams['max_epochs']

    final_results_summary = []

    # 3. 加载数据, 传入 batch_size
    loaders, model_inputs = load_and_preprocess_data(cfg, batch_size)

    # for seed in seeds:
    #     print("\n" + "="*30 + f" 开始执行，随机种子: {seed} " + "="*30)
    #     set_seed(seed)

    #     # 4. 生成随机超参数组合 (使用来自 JSON 的 search_space)
    #     param_combinations = []
    #     for _ in range(N_TRIALS):
    #         params = {
    #             'n_blocks': random.choice(search_space['n_blocks']),
    #             'ffn_d_hidden': random.choice(search_space['ffn_d_hidden']),
    #             'residual_dropout': random.uniform(*search_space['residual_dropout']),
    #             'attention_dropout': random.uniform(*search_space['attention_dropout']),
    #             'ffn_dropout': random.uniform(*search_space['ffn_dropout']),
    #             'd_token': D_TOKEN, 
    #             'learning_rate': LEARNING_RATE, 
    #             'weight_decay': WEIGHT_DECAY,
    #         }
    #         param_combinations.append(params)
        
    #     # 5. 阶段一：搜索
    #     best_params = search_for_best_params(
    #         param_combinations, cfg, seed, loaders, model_inputs, device
    #     )
        
    #     # [!] 6. 阶段二：训练, 传入 patience 和 max_epochs
    #     final_model = train_final_model(
    #         best_params, cfg, seed, loaders, model_inputs, device,
    #         patience=patience, max_epochs=max_epochs
    #     )
        
    #     # 7. 阶段三：评估
    #     print("\n" + "-"*10 + f" [种子 {seed}] 阶段三：在测试集上进行最终评估 " + "-"*10)
    #     metrics_dict, result_line = evaluate_final_model(
    #         final_model, loaders['test'], model_inputs['task'], device
    #     )
        
    #     result_dict = {
    #         'seed': seed, 
    #         'best_params': best_params, 
    #         'result_line': result_line,
    #         **metrics_dict
    #     }
    #     final_results_summary.append(result_dict)

    manual_best_params = {
        'n_blocks': 3, 
        'ffn_d_hidden': 64, 
        'residual_dropout': 0.181130119829332, 
        'attention_dropout': 0.437874063436074, 
        'ffn_dropout': 0.18396799001208675, 
        'd_token': 192, 
        'learning_rate': 0.0001, 
        'weight_decay': 1e-05
    }
    
    print("!!! 检测到手动模式：跳过阶段一搜索，直接使用已知最佳参数进行训练 !!!")

    for seed in seeds:
        print("\n" + "="*30 + f" 开始执行，随机种子: {seed} " + "="*30)
        set_seed(seed)

        # 🔴 2. 注释掉阶段一搜索代码
        # param_combinations = ...
        # best_params = search_for_best_params(...)
        
        # 直接赋值
        best_params = manual_best_params

        # 🔴 3. 执行阶段二 (这个必须重跑，因为模型文件之前被自动删除了)
        # 这次跑会很快，因为只需要训练这一个模型，而且根据日志它在第10个epoch就早停了
        final_model = train_final_model(
            best_params, cfg, seed, loaders, model_inputs, device,
            patience=patience, max_epochs=max_epochs
        )
        
        # 🔴 4. 执行阶段三 (这里是你报错的地方，现在已经修复了)
        print("\n" + "-"*10 + f" [种子 {seed}] 阶段三：在测试集上进行最终评估 " + "-"*10)
        metrics_dict, result_line = evaluate_final_model(
            cfg, final_model, loaders['test'], model_inputs['task'], device
        )
        
        result_dict = {
            'seed': seed, 
            'best_params': best_params, 
            'result_line': result_line,
            **metrics_dict
        }
        final_results_summary.append(result_dict)

    # --- 7. 最终总结 ---
    print("\n\n" + "="*30 + " 所有实验最终总结 " + "="*30)
    
    output_file_path = 'result/fttrans.txt'
    print(f"准备将结果写入到: {output_file_path}")

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, 'a') as f:
        f.write(f"--- 实验配置 (来自 config.yaml): {cfg.target} ---\n")
        f.write(OmegaConf.to_yaml(cfg)) # 写入数据配置
        f.write("\n--- 模型配置 (来自 ftt_model_config.json) ---\n")
        f.write(json.dumps(model_config, indent=2)) # [!] 写入模型配置
        f.write("\n\n" + "="*30 + " 所有实验最终总结 " + "="*30 + "\n")
        
        for final_result in final_results_summary:
            result_line = f"种子: {final_result['seed']} | {final_result['result_line']}"
            print(result_line)
            f.write(result_line + "\n")

        params_header = f"\n最佳参数的例子 (来自最后一个种子 {final_results_summary[-1]['seed']}):"
        params_details = str(final_results_summary[-1]['best_params'])
        
        print(params_header)
        print(params_details)
        f.write(params_header + "\n")
        f.write(params_details + "\n")
        f.write("="*80 + "\n")

    print(f"\n结果已成功写入到文件: {output_file_path}")


if __name__ == "__main__":
    main()