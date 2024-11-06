env="Pursuit"
map="4m"
algo="mappo"

exp="TeamBTrainSEED20(GameOnlyRwd)Long"
# dir_path="/home/marmot/Pamela/results/Pursuit/4m/mappo/TeamATrainSEED10(GameOnlyRwd)Long/run7/wandb/run-20241008_081952-v5p329aw/files" # CHANGE THIS FOR Each policy model
# dir_path="/home/marmot/pth/results/Pursuit/4m/mappo/TeamATrainSEED10Catch3/run2/wandb/run-20241010_234050-9bgzbol7/files"
# dir_path="/home/marmot/pth/results/Pursuit/4m/mappo/TeamBTrainSEED20Catch3/run1/wandb/run-20241010_234150-cs6i566d/files"
# dir_path="/home/marmot/pth/results/Pursuit/4m/mappo/TeamATrainSEED10(GameOnlyRwd)Long/run2/wandb/run-20241010_152034-4gnny8dm/files"
dir_path="/home/marmot/pth/results/Pursuit/4m/mappo/TeamBTrainSEED20(GameOnlyRwd)Long/run2/wandb/run-20241010_152109-gianswgj/files"
# dir_path="/home/marmot/Pamela/results/Pursuit/4m/mappo/TeamATrainSEED10(GameOnlyRwd)LongShared/run3/wandb/run-20241008_182110-4tvy8et3/files"
seed=20
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo ${dir_path}
for i in 50 100
do
    checkpoint="Policy_Episode${i}.0M"
    file_path="${dir_path}/${checkpoint}"
    echo "Loading for ${file_path}"
    expFormalName="${exp}-${i}M-Seed${seed}"
    echo "experiment is ${expFormalName}"
    CUDA_VISIBLE_DEVICES=0 python ./train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${expFormalName} \
    --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 500 \
    --group_social_influence "OnlyGameRwd" \
    --num_env_steps 50000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 --eval_interval 500  --use_wandb  --no_train_policy --model_dir ${file_path} --si_loss_type 'kl' --minimal_gifs

done