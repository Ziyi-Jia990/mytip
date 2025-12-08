#!/bin/bash

# --- 1. 配置 ---

# 您的候选学习率列表 (用空格隔开)
LR_LIST="1e-3 1e-4 1e-5"
MODEL_LIST="MUL MAX Concat DAFT image"



# 微调的最大 Epochs 数 (早停法会自动处理)
MAX_EPOCHS_FINETUNE=500

# --- 2. 循环执行 ---

echo "开始学习率搜索 (Hydra 模式)..."

# 遍历列表中的每一个学习率
for MODEL in $MODEL_LIST
do
    for LR in $LR_LIST
    do
        # 为这次运行创建一个唯一的实验名称
        # 您的 run.py 会读取这个 exp_name 并用它来创建日志文件夹
        RUN_NAME="anime_${MODEL}_lr_${LR}"
        CONFIG_NAME="config_anime_${MODEL}.yaml"
        
        echo "-----------------------------------------------------"
        echo "开始运行: LR = $LR"
        echo "实验名称: $RUN_NAME"
        echo "-----------------------------------------------------"
        
        # 完整复刻您的运行指令，并添加 Hydra 覆盖
        CUDA_VISIBLE_DEVICES=0 python -u run.py \
            --config-name $CONFIG_NAME \
            exp_name=$RUN_NAME \
            max_epochs=$MAX_EPOCHS_FINETUNE \
            lr_eval=$LR \
            lr=$LR \
            use_wandb=False
            
        echo "完成运行: LR = $LR"
    done
done 

echo "所有学习率搜索已完成。"