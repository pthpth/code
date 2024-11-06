env="StarCraft2"
map="4m"
teamBMap="4m_vs_5m"
algo="mappo"

seed=7
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

# There is no 'only use argmax action' in the above command. Since we are doing training
exp="AdhocTrain_SMAC-4M_4MV5VDiffEnv-MAPPO-MultiplyRwd-50MPolicyVsUntrainedB1"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --teamB_map_name ${teamBMap} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--group_social_influence "MultiplyGameSocialRwd" \
--num_env_steps 5000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --eval_interval 100 --run_mixed_population \
--policy_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M" \
--si_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-50M-Seed6-Policy50M/social_influence_ts50.0M" \
 --percent_team_A_agents 75 --group_social_influence_factor 0.5 # --group_social_influence_factor 0.5


###########SimplyTrained
# exp="AdhocTrain_SMAC-4M_4MV5VDiffEnv-MAPPO(ComboModRwd)-High(Penalty)"
exp="AdhocTrain_SMAC-4M_4MV5VDiffEnv-MAPPO-MultiplyRwd-50MPolicyVsUntrainedB2"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --teamB_map_name ${teamBMap} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--group_social_influence "MultiplyGameSocialRwd" \
--num_env_steps 5000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --eval_interval 100 --run_mixed_population \
--policy_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M" \
--si_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-50M-Seed6-Policy50M/social_influence_ts50.0M" \
 --percent_team_A_agents 50 --group_social_influence_factor 0.5 # --group_social_influence_factor 0.5


## 4m Seed 6
# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M
# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode50.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-50M-Seed6-Policy50M/social_influence_ts50.0M

## 4m_vs_5m Seed 3
# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode50.0M
# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode50.0M/SMAC-LOAD4m_vs_5m-TrainSI-MAPPO-Steps50M-50M-Policy50M/social_influence_ts50.0M