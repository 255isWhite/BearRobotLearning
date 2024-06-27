bs=64
ws=2
export WANDB_API_KEY=0d4d8e6f87ec9508a673bb4f0d117bf6a79a9945
torchrun --standalone --nnodes=1 --nproc_per_node=6 train_example/train_diffusion_policy_decisionNCE_example_libero_copy.py \
    --seed 42 \
    --dataset 'libero_goal' \
    --algo_name 'dnce diffusion visual motor' \
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
    --val_freq 10000000 \
    --eval_freq 100000000 \
    --resume None \
    --wandb True \
    --steps 50000 \
    --save True \
    --save_freq 10000 \
    --T 25 \
    --save_path ../experiments/libero/libero_goal/diffusion_dnce/0627test01 \
    --log_path ../experiments/libero/libero_goal/diffusion_dnce/0627test01 \
    --port 2052 \