env="StarCraft2"
map="4m"
teamBMap="4m_vs_5m"
algo="mappo"

dir_path_A="../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)"
dir_path_B="../../official_saved_models/4m_vs_5m/SMAC-4m_vs_5m-MAPPO-SuperLong50M(Seed3_Run1_Ser1)"

si_train_stage=5
policy_train_stage=10

if [ $# -lt 1 ]; then
    echo "Usage: $0 <RewardType>"
    exit 1
fi

reward_type="$1"
exppre="${1}_TESTER"

echo "Starting script for ${exppre} we do train SI and policy here"

for seed in {30..50..5}
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


echo "Finished check for all files exist ${exppre}"

seed=30

# First pair: si_train_stage = 50, policy_train_stage = 10
exp="${exppre}-SI${si_train_stage}M-Policy${policy_train_stage}M-Seed${seed}"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

checkpoint="Policy_Episode${policy_train_stage}.0M"
file_path_A="${dir_path_A}/${checkpoint}"
file_path_B="${dir_path_B}/${checkpoint}"
echo "Loading for ${file_path_A} and ${file_path_B}"

si_model_path_A="${file_path_A}/SI/social_influence_ts${si_train_stage}.0M"
si_model_path_B="${file_path_B}/SI/social_influence_ts${si_train_stage}.0M"

CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
--map_name ${map} --teamB_map_name ${teamBMap} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 500 \
--group_social_influence ${reward_type} \
--num_env_steps 10000 --ppo_epoch 15 --use_value_active_masks --use_eval --eval_episodes 32  --cuda --eval_interval 10 --run_mixed_population \
--policy_model_dir_teamA ${file_path_A} \
--policy_model_dir_teamB ${file_path_B} \
--si_model_dir_teamA ${si_model_path_A} \
--si_model_dir_teamB ${si_model_path_B} --percent_team_A_agents 50 --group_social_influence_factor 0.7 

status=$?
if ! (exit $status); then
    echo "Run failed"
    exit 1
fi


