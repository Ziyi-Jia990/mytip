#!/bin/bash

# --- 1. 配置 ---

# 您的候选学习率列表 (用空格隔开)
LR_LIST="1e-3 1e-4 1e-5"
# LR_LIST="1e-3"

# 您的预训练 checkpoint 路径
# PRETRAINED_CKPT="/home/jiazy/mytip/results/runs/multimodal/anime_2022_MMCL_anime_1204_1154/checkpoint_last_epoch_499.ckpt"
PRETRAINED_CKPT='/home/jiazy/mytip/results/runs/multimodal/rr_MMCL_rr_1214_0419/checkpoint_last_epoch_00.ckpt'

# 您的 Hydra 配置文件名
CONFIG_NAME="config_rr_MMCL.yaml"

# 微调的最大 Epochs 数 (早停法会自动处理)
MAX_EPOCHS_FINETUNE=500

# --- 2. 循环执行 ---

echo "开始学习率搜索 (Hydra 模式)..."

# 遍历列表中的每一个学习率
for LR in $LR_LIST
do
    # 为这次运行创建一个唯一的实验名称
    # 您的 run.py 会读取这个 exp_name 并用它来创建日志文件夹
    RUN_NAME="rr_MMCL_lr_${LR}"
    
    echo "-----------------------------------------------------"
    echo "开始运行: LR = $LR"
    echo "实验名称: $RUN_NAME"
    echo "-----------------------------------------------------"
    
    # 完整复刻您的运行指令，并添加 Hydra 覆盖
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --config-name $CONFIG_NAME \
        pretrain=False \
        evaluate=True \
        checkpoint=$PRETRAINED_CKPT \
        exp_name=$RUN_NAME \
        max_epochs=$MAX_EPOCHS_FINETUNE \
        lr_eval=$LR \
        use_wandb=False 
        
    echo "完成运行: LR = $LR"
done

echo "所有学习率搜索已完成。"
