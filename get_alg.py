# import torch

# # checkpoint 文件路径
# checkpoint_path = "/mnt/hdd/jiazy/checkpoints/dvm/Max/checkpoint_best_acc.ckpt"

# # 加载 checkpoint
# checkpoint = torch.load(checkpoint_path, map_location="cpu")

# # 检查是否存在 'algorithm_name' 键
# if "algorithm_name" in checkpoint:
#     print("algorithm_name:", checkpoint["algorithm_name"])
# else:
#     print("❌ 'algorithm_name' not found in checkpoint keys.")
#     print("Available keys:", checkpoint.keys())

# print(checkpoint['hyper_parameters']['finetune_ensemble'])

import torch
from omegaconf import OmegaConf, open_dict

# --- 1. 请修改这里的路径 ---
# 替换成你的原始 checkpoint 路径
CHECKPOINT_PATH_IN = '/mnt/hdd/jiazy/checkpoints/dvm/Max/checkpoint_best_acc.ckpt' 
# 修复后的新路径（注意：这里与输入路径相同，将会覆盖原始文件）
CHECKPOINT_PATH_OUT = "/mnt/hdd/jiazy/checkpoints/dvm/Max/checkpoint_best_acc.ckpt" 

# --------------------------

print(f"正在加载 checkpoint: {CHECKPOINT_PATH_IN}")
# 加载到 cpu，避免占用 GPU
checkpoint = torch.load(CHECKPOINT_PATH_IN, map_location='cpu')

# 访问 hparams (它是一个 OmegaConf 对象)
hparams = checkpoint['hyper_parameters']
print("------------------------------------------")
print("原始 hparams (部分内容):")
print(f"  target: {hparams.get('target')}")
print(f"  task: {hparams.get('task')}")
print(f"  data_train_imaging: {hparams.get('data_train_imaging')}")
print("------------------------------------------")


# 使用 open_dict 来允许修改 OmegaConf
try:
    with open_dict(hparams):
        print("正在使用 YAML 内容更新 hparams...")
        
        # --- 自动填入您提供的 YAML 修改内容 ---
        hparams.target = 'dvm'
        hparams.task = 'classification'
        hparams.data_base = '/mnt/hdd/jiazy/DVM-Car/features'
        
        hparams.num_classes = 286
        hparams.weights = None # YAML 中 'weights:' 为空，这里设为 None
        hparams.live_loading = True
        hparams.delete_segmentation = False
        hparams.balanced_accuracy = False
        hparams.eval_metric = 'acc'
        hparams.data_orig = None # YAML 中 'data_orig:' 为空
        hparams.low_data_splits = ['']

        # num of features
        hparams.num_cat = 4
        hparams.num_con = 13

        hparams.labels_train = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_train_all_views.pt'
        hparams.labels_val = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_val_all_views.pt'

        hparams.data_train_imaging = '/mnt/hdd/jiazy/DVM-Car/features/train_paths_all_views.pt'
        hparams.data_val_imaging = '/mnt/hdd/jiazy/DVM-Car/features/val_paths_all_views.pt'

        hparams.data_train_tabular = '/mnt/hdd/jiazy/DVM-Car/features/dvm_features_train_noOH_all_views_physical_jittered_50.csv'
        hparams.data_val_tabular = '/mnt/hdd/jiazy/DVM-Car/features/dvm_features_val_noOH_all_views_physical_jittered_50.csv'

        hparams.field_lengths_tabular = '/mnt/hdd/jiazy/DVM-Car/features/tabular_lengths_all_views_physical.pt'

        hparams.data_train_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/dvm_features_train_noOH_all_views_physical_jittered_50.csv'
        hparams.labels_train_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_train_all_views.pt'

        hparams.data_val_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/dvm_features_val_noOH_all_views_physical_jittered_50.csv'
        hparams.labels_val_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_val_all_views.pt'

        hparams.data_test_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/dvm_features_test_noOH_all_views_physical_jittered_50.csv'
        hparams.labels_test_eval_tabular = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_test_all_views.pt'

        hparams.data_train_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/train_paths_all_views.pt'
        hparams.labels_train_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_train_all_views.pt'

        hparams.data_val_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/val_paths_all_views.pt'
        hparams.labels_val_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_val_all_views.pt'

        hparams.data_test_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/test_paths_all_views.pt'
        hparams.labels_test_eval_imaging = '/mnt/hdd/jiazy/DVM-Car/features/labels_model_all_test_all_views.pt'
        
        print("Hparams 更新完成。")

except Exception as e:
    print(f"修改 hparams 出错: {e}")
    print("请检查你的 checkpoint 结构是否正确。")
    exit()

print("------------------------------------------")
print("修改后 hparams (部分内容):")
print(f"  target: {hparams.get('target')}")
print(f"  task: {hparams.get('task')}")
print(f"  data_train_imaging: {hparams.get('data_train_imaging')}")
print("------------------------------------------")


# 4. 保存新的 (修复后的) checkpoint
try:
    torch.save(checkpoint, CHECKPOINT_PATH_OUT)
    print(f"已成功保存修复后的文件到: {CHECKPOINT_PATH_OUT}")
    print("\n现在请在你的评估命令中改用这个新的 .ckpt 文件。")
except Exception as e:
    print(f"保存新文件时出错: {e}")

# # --- 2. 请定义正确的值 ---
# DATA_BASE = '/mnt/hdd/jiazy/DVM-Car/features'
# # --------------------------

# print(f"正在加载 checkpoint: {CHECKPOINT_PATH_IN}")
# # 加载到 cpu，避免占用 GPU
# checkpoint = torch.load(CHECKPOINT_PATH_IN, map_location='cpu')

# # 访问 hparams (它是一个 OmegaConf 对象)
# hparams = checkpoint['hyper_parameters']
# print("------------------------------------------")


# # 使用 open_dict 来允许修改 OmegaConf
# try:
#     with open_dict(hparams):
#         hparams.data_base = DATA_BASE
# except Exception as e:
#     print(f"修改 hparams 出错: {e}")
#     print("请检查你的 checkpoint 结构是否正确。")
#     exit()

# print(f"修改后 hparams: num_cat={hparams.num_cat}, num_con={hparams.num_con}")
# print("------------------------------------------")

# # 4. 保存新的 (修复后的) checkpoint
# try:
#     torch.save(checkpoint, CHECKPOINT_PATH_OUT)
#     print(f"已成功保存修复后的文件到: {CHECKPOINT_PATH_OUT}")
#     print("\n现在请在你的评估命令中改用这个新的 .ckpt 文件。")
# except Exception as e:
#     print(f"保存新文件时出错: {e}")