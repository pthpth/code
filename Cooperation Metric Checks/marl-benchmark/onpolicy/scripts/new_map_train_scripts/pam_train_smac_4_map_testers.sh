env="StarCraft2"
algo="mappo"
seed=1



map="4m"
exp="SMAC-4M-MAPPO-TESTER"

# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
# for seed in `seq ${seed_max}`;
# do
# echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 15000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --minimal_gifs --eval_interval 5 --si_loss_type 'kl'
# done


map="4m_vs_5m"
exp="SMAC-4m_vs_5m-MAPPO-TESTER"

# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--num_env_steps 15000 --ppo_epoch 10 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --minimal_gifs --eval_interval 5 --si_loss_type 'kl'

