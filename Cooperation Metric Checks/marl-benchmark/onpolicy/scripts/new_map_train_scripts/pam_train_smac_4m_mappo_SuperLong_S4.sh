env="StarCraft2"
map="4m"
algo="mappo"

exp="SMAC-4M-MAPPO-Steps50M"

# seed=3
# seed_max=2
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

# for seed in `seq ${seed_max}`;
for seed in {1..4..2}
do
    echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 50000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --minimal_gifs --eval_interval 100 --si_loss_type 'kl'
done
