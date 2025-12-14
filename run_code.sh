# TIP
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_TIP exp_name=pretrain_pneumonia_TIP_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_TIP exp_name=pretrain_pneumonia_TIP_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_TIP exp_name=pretrain_pneumonia_TIP_2024 use_wandb=False seed=2024


CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_TIP exp_name=pretrain_los_TIP_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_TIP exp_name=pretrain_los_TIP_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_TIP exp_name=pretrain_los_TIP_2024 use_wandb=False seed=2024


CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_TIP exp_name=pretrain_rr_TIP_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_TIP exp_name=pretrain_rr_TIP_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_TIP exp_name=pretrain_rr_TIP_2024 use_wandb=False seed=2024

# MMCL
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MMCL exp_name=pretrain_pneumonia_MMCL_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MMCL exp_name=pretrain_pneumonia_MMCL_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pneumonia_MMCL exp_name=pretrain_pneumonia_MMCL_2024 use_wandb=False seed=2024


CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MMCL exp_name=pretrain_los_MMCL_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MMCL exp_name=pretrain_los_MMCL_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_los_MMCL exp_name=pretrain_los_MMCL_2024 use_wandb=False seed=2024


CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MMCL exp_name=pretrain_rr_MMCL_2022 use_wandb=False seed=2022
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MMCL exp_name=pretrain_rr_MMCL_2023 use_wandb=False seed=2023
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_rr_MMCL exp_name=pretrain_rr_MMCL_2024 use_wandb=False seed=2024