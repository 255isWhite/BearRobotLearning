bs=64
ws=2
path=../experiments/libero/libero_goal/minibc/0626test02

export WANDB_API_KEY=0d4d8e6f87ec9508a673bb4f0d117bf6a79a9945
torchrun --standalone --nnodes=1 --nproc_per_node=6 train_example/train_mini_bc.py \
    --seed 42 \
    --dataset 'libero_goal' \
    --algo_name 'Mini BC with pretrained DecisionNCE-T' \
    --ddp True \
    --mm_encoder DecisionNCE-T \
    --ft_mmencoder False \
    --film_fusion False \
    --ac_num 6 \
    --norm minmax \
    --discretize_actions False \
    --encode_s False \
    --encode_a False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0003 \
    --val_freq 20000000000 \
    --eval_freq 200000000000 \
    --resume None \
    --wandb True \
    --steps 20000 \
    --save True \
    --save_freq 10000 \
    --save_path $path \
    --log_path $path \
    --port 2052 