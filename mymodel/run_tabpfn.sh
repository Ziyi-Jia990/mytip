for seed in 2022 2023 2024
do
    echo "Running with seed=$seed"
    CUDA_VISIBLE_DEVICES=1 python eval_tabpfn.py seed=$seed dataset=dvm_all_server
done
