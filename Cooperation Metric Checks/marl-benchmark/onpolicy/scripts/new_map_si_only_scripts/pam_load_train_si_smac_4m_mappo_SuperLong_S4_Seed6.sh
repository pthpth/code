env="StarCraft2"
map="4m"
algo="mappo"

exp="SMAC-LOAD4M-TrainSI-MAPPO-Steps50M"
dir_path="../../official_saved_models/4m/SMAC-4M-MAPPO-Steps50M(Seed6_Run1_Ser1)"
seed=6

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
for i in {10..50..20}
do
    checkpoint="Policy_Episode${i}.0M"
    file_path="${dir_path}/${checkpoint}"
    echo "Loading for ${file_path}"
    expFormalName="${exp}-${i}M-Seed${seed}"
    echo "experiment is ${expFormalName}"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${expFormalName} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
    --num_env_steps 50000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 --eval_interval 100 --cuda --use_wandb  --no_train_policy --model_dir ${file_path} --si_loss_type 'kl' --only_use_argmax_action --minimal_gifs

done

