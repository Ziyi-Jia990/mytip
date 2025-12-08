# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_MUL.yaml \
#     exp_name=pawpularity_MUL_2023 \
#     max_epochs=500 \
#     seed=2023 \
#     lr_eval=1e-4 \
#     lr=1e-4 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_MUL.yaml \
#     exp_name=pawpularity_MUL_2024 \
#     max_epochs=500 \
#     seed=2024 \
#     lr_eval=1e-4 \
#     lr=1e-4 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_MAX.yaml \
#     exp_name=pawpularity_MAX_2023 \
#     max_epochs=500 \
#     seed=2023 \
#     lr_eval=1e-5 \
#     lr=1e-5 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_MAX.yaml \
#     exp_name=pawpularity_MAX_2024 \
#     max_epochs=500 \
#     seed=2024 \
#     lr_eval=1e-5 \
#     lr=1e-5 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_Concat.yaml \
#     exp_name=pawpularity_Concat_2023 \
#     max_epochs=500 \
#     seed=2023 \
#     lr_eval=1e-3 \
#     lr=1e-3 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_Concat.yaml \
#     exp_name=pawpularity_Concat_2024 \
#     max_epochs=500 \
#     seed=2024 \
#     lr_eval=1e-3 \
#     lr=1e-3  \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_DAFT.yaml \
#     exp_name=pawpularity_DAFT_2023 \
#     max_epochs=500 \
#     seed=2023 \
#     lr_eval=1e-5 \
#     lr=1e-5 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_DAFT.yaml \
#     exp_name=pawpularity_DAFT_2024 \
#     max_epochs=500 \
#     seed=2024 \
#     lr_eval=1e-5 \
#     lr=1e-5  \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_image.yaml \
#     exp_name=pawpularity_image_2023 \
#     max_epochs=500 \
#     seed=2023 \
#     lr_eval=1e-5 \
#     lr=1e-5 \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --config-name config_pawpularity_image.yaml \
#     exp_name=pawpularity_image_2024 \
#     max_epochs=500 \
#     seed=2024 \
#     lr_eval=1e-5 \
#     lr=1e-5  \
#     use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_TIP exp_name=pawpularity_2022_TIP seed=2022 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_TIP exp_name=pawpularity_2023_TIP seed=2023 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_TIP exp_name=pawpularity_2024_TIP seed=2024 use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_MMCL exp_name=pawpularity_2022_MMCL seed=2022 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_MMCL exp_name=pawpularity_2023_MMCL seed=2023 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_pawpularity_MMCL exp_name=pawpularity_2024_MMCL seed=2024 use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#         --config-name config_pawpularity_MMCL \
#         pretrain=False \
#         evaluate=True \
#         checkpoint=/home/jiazy/mytip/results/runs/multimodal/pawpularity_2023_MMCL_pawpularity_1203_0819/checkpoint_last_epoch_499.ckpt \
#         exp_name=pawpularity_2023_MMCL \
#         max_epochs=500 \
#         lr_eval=1e-3 \
#         use_wandb=False \
#         seed=2023

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#         --config-name config_pawpularity_MMCL \
#         pretrain=False \
#         evaluate=True \
#         checkpoint=/home/jiazy/mytip/results/runs/multimodal/pawpularity_2024_MMCL_pawpularity_1203_0924/checkpoint_last_epoch_499.ckpt \
#         exp_name=pawpularity_2024_MMCL \
#         max_epochs=500 \
#         lr_eval=1e-3 \
#         use_wandb=False \
#         seed=2024

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#         --config-name config_pawpularity_TIP \
#         pretrain=False \
#         evaluate=True \
#         checkpoint=/home/jiazy/mytip/results/runs/multimodal/pawpularity_2023_TIP_pawpularity_1203_0311/checkpoint_last_epoch_499.ckpt \
#         exp_name=pawpularity_2023_TIP \
#         max_epochs=500 \
#         lr_eval=1e-3 \
#         use_wandb=False \
#         seed=2023

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#         --config-name config_pawpularity_TIP \
#         pretrain=False \
#         evaluate=True \
#         checkpoint=/home/jiazy/mytip/results/runs/multimodal/pawpularity_2024_TIP_pawpularity_1203_0510/checkpoint_last_epoch_499.ckpt \
#         exp_name=pawpularity_2024_TIP \
#         max_epochs=500 \
#         lr_eval=1e-3 \
#         use_wandb=False \
#         seed=2024

# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_TIP exp_name=anime_2022_TIP seed=2022 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_TIP exp_name=anime_2023_TIP seed=2023 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_TIP exp_name=anime_2024_TIP seed=2024 use_wandb=False

# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_MMCL exp_name=anime_2022_MMCL seed=2022 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_MMCL exp_name=anime_2023_MMCL seed=2023 use_wandb=False
# CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_anime_MMCL exp_name=anime_2024_MMCL seed=2024 use_wandb=False

# bash search_lr.sh

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_MUL.yaml \
    exp_name=anime_MUL_2023 \
    max_epochs=500 \
    seed=2023 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_MUL.yaml \
    exp_name=anime_MUL_2024 \
    max_epochs=500 \
    seed=2024 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_MAX.yaml \
    exp_name=anime_MAX_2023 \
    max_epochs=500 \
    seed=2023 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_MAX.yaml \
    exp_name=anime_MAX_2024 \
    max_epochs=500 \
    seed=2024 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_Concat.yaml \
    exp_name=anime_Concat_2023 \
    max_epochs=500 \
    seed=2023 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_Concat.yaml \
    exp_name=anime_Concat_2024 \
    max_epochs=500 \
    seed=2024 \
    lr_eval=1e-3 \
    lr=1e-3  \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_DAFT.yaml \
    exp_name=anime_DAFT_2023 \
    max_epochs=500 \
    seed=2023 \
    lr_eval=1e-4 \
    lr=1e-4 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_DAFT.yaml \
    exp_name=anime_DAFT_2024 \
    max_epochs=500 \
    seed=2024 \
    lr_eval=1e-4 \
    lr=1e-4  \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_image.yaml \
    exp_name=anime_image_2023 \
    max_epochs=500 \
    seed=2023 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --config-name config_anime_image.yaml \
    exp_name=anime_image_2024 \
    max_epochs=500 \
    seed=2024 \
    lr_eval=1e-3 \
    lr=1e-3 \
    use_wandb=False

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --config-name config_anime_MMCL \
        pretrain=False \
        evaluate=True \
        checkpoint=/home/jiazy/mytip/results/runs/multimodal/anime_2023_MMCL_anime_1204_1327/checkpoint_last_epoch_499.ckpt \
        exp_name=anime_2023_MMCL \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2023

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --config-name config_anime_MMCL \
        pretrain=False \
        evaluate=True \
        checkpoint=/home/jiazy/mytip/results/runs/multimodal/anime_2024_MMCL_anime_1204_1508/checkpoint_last_epoch_499.ckpt \
        exp_name=anime_2024_MMCL \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2024

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --config-name config_anime_TIP \
        pretrain=False \
        evaluate=True \
        checkpoint=/home/jiazy/mytip/results/runs/multimodal/anime_2023_TIP_anime_1203_1800/checkpoint_last_epoch_499.ckpt \
        exp_name=anime_2023_TIP \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2023

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --config-name config_anime_TIP \
        pretrain=False \
        evaluate=True \
        checkpoint=/home/jiazy/mytip/results/runs/multimodal/anime_2024_TIP_anime_1204_0636/checkpoint_last_epoch_499.ckpt \
        exp_name=anime_2024_TIP \
        max_epochs=500 \
        lr_eval=1e-3 \
        use_wandb=False \
        seed=2024