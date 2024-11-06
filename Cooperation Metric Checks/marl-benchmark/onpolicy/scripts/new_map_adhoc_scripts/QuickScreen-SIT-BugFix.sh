env="StarCraft2"
map="4m"
teamBMap="4m_vs_5m"
algo="mappo"

seed=15
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"

exppre="AdhocTrainStages_SMACDiffEnv-MAPPO(RBLens)"

dir_path_A="../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)"
dir_path_B="../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)"
for seed in {30..40..5}
do
    for si_train_stage in {10..50..40}
    do
        for policy_train_stage in {10..50..40}
        do  
            exp="${exppre}-SI${si_train_stage}M-Policy${policy_train_stage}M-Seed${seed}"
            # echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

            checkpoint="Policy_Episode${policy_train_stage}.0M"
            file_path_A="${dir_path_A}/${checkpoint}"
            file_path_B="${dir_path_B}/${checkpoint}"
            # echo "Loading for ${file_path_A} and ${file_path_B}"

            si_model_path_A="${file_path_A}/SI/social_influence_ts${si_train_stage}.0M"
            si_model_path_B="${file_path_B}/SI/social_influence_ts${si_train_stage}.0M"

			# Debug prints
			# echo "Checking file A: \"${si_model_path_A}\""
			ls -l "${si_model_path_A}" &>/dev/null || echo "File \"${si_model_path_A}\" not found"
			# echo "Checking file B: \"${si_model_path_B}\""
			ls -l "${si_model_path_B}" &>/dev/null || echo "File \"${si_model_path_B}\" not found" 

        done
    done
done


echo "Finished check for all files exist"


exppre="AdhocTrainStages_SIComboPenalty_(TrainSI)(RBLens)_SMACDiffEnv-MAPPO"

seed=40

# Second pair: si_train_stage = 10, policy_train_stage = 50
si_train_stage=10
policy_train_stage=50

exp="${exppre}-SI${si_train_stage}M-Policy${policy_train_stage}M-Seed${seed}"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

checkpoint="Policy_Episode${policy_train_stage}.0M"
file_path_A="${dir_path_A}/${checkpoint}"
file_path_B="${dir_path_B}/${checkpoint}"
echo "Loading for ${file_path_A} and ${file_path_B}"

si_model_path_A="${file_path_A}/SI/social_influence_ts${si_train_stage}.0M"
si_model_path_B="${file_path_B}/SI/social_influence_ts${si_train_stage}.0M"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --teamB_map_name ${teamBMap} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
--group_social_influence "CombineSocialGameRwd_withPenalty" \
--num_env_steps 3000000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --use_wandb --eval_interval 20 --run_mixed_population \
--policy_model_dir_teamA ${file_path_A} \
--policy_model_dir_teamB ${file_path_B} \
--si_model_dir_teamA ${si_model_path_A} \
--si_model_dir_teamB ${si_model_path_B} --percent_team_A_agents 50 --group_social_influence_factor 0.7 
