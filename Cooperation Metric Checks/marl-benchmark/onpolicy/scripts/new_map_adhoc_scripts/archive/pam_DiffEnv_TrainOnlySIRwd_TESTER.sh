env="StarCraft2"
map="4m"
teamBMap="4m_vs_5m"
algo="mappo"

seed=10
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

###########SimplyTrained
exp="TESTER-Adhoc_Training_SMAC-4M_4MV5VDiffEnv-MAPPO(SIRwd)"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --teamB_map_name ${teamBMap} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--group_social_influence "OnlySocialRwd" \
--num_env_steps 30000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --eval_interval 2 --run_mixed_population --policy_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M" --policy_model_dir_teamB "../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M" --si_model_dir_teamA "../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M" --si_model_dir_teamB "../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4m_vs_5m-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M" --percent_team_A_agents 50

# There is no 'only use argmax action' in the above command. Since we are doing training

# TODO Change the model paths here and 

# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M
# ../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4M-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M

# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M
# ../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)/Policy_Episode10.0M/SMAC-LOAD4m_vs_5m-TrainSI-MAPPO-Steps50M-10M-Crashed/social_influence_ts10.0M