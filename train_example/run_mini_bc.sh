bs=64
ws=2
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=0d4d8e6f87ec9508a673bb4f0d117bf6a79a9945
torchrun --standalone --nnodes=1 --nproc-per-node=1 train_diffusion_policy_decisionNCE_example_libero.py \
    --seed 42 \
    --dataset 'libero_goal' \
    --algo_name 'Mini BC with pretrained DecisionNCE-T' \
    --ddp False \
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
    --val_freq 10000000 \
    --eval_freq 25000 \
    --resume None \
    --wandb True \
    --steps 20000 \
    --save True \
    --save_freq 10000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/minibc/0622test01 \
    --log_path ../experiments/libero/libero_goal/minibc/0622test01  \
    --port 2052 \