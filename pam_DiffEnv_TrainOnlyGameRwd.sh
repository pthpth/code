env="Pursuit"
map="4m"
# teamBMap="4m_vs_5m"
algo="mappo"
n_units=8
n_enemies=30
seed=10
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

###########SimplyTrained
exp="TeamA(GameOnlyRwd)IndRewards"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python ./train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--n_enemies ${n_enemies} --n_units ${n_units} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 500 \
--group_social_influence "OnlyGameRwd" \
--n_eval_rollout_threads 4 \
--num_env_steps 50000000 --ppo_epoch 15  --use_wandb  --si_loss_type 'kl' --use_eval --use_value_active_masks --minimal_gifs

# There is no 'only use argmax action' in the above command. Since we are doing training

# TODO Change the model paths here and 

# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M
# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M

# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M
# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4m_vs_5m-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M
