env="Pursuit"
map="4m"
teamB_map_name="4m"
algo="mappo"
policy_model_dir_TeamA="/home/marmot/pth/results/Pursuit/4m/mappo/TeamATrainSEED10(GameOnlyRwd)Long/run2/wandb/run-20241010_152034-4gnny8dm/files/Policy_Episode20.0M"
policy_model_dir_TeamB="/home/marmot/pth/results/Pursuit/4m/mappo/TeamBTrainSEED20(GameOnlyRwd)Long/run2/wandb/run-20241010_152109-gianswgj/files/Policy_Episode20.0M"
si_model_dir_TeamB="/home/marmot/pth/results/Pursuit/4m/mappo/TeamBTrainSEED20(GameOnlyRwd)Long/run2/wandb/run-20241010_152109-gianswgj/files/social_influence_ts20.0M"
si_model_dir_TeamA="/home/marmot/pth/results/Pursuit/4m/mappo/TeamATrainSEED10(GameOnlyRwd)Long/run2/wandb/run-20241010_152034-4gnny8dm/files/social_influence_ts20.0M"


reward="CombineSocialGameRwd"
exp="TeamAandBTrain0${reward}"
seed=10
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
echo ${dir_path}
expFormalName="${exp}-${i}M-Seed${seed}"
echo "experiment is ${expFormalName}"
CUDA_VISIBLE_DEVICES=0 python ./train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${expFormalName} \
--map_name ${map} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 500 \
--group_social_influence ${reward} --teamB_map_name ${teamB_map_name} \
---group_social_influence_factor 1.0 \
--run_mixed_population --policy_model_dir_teamA ${policy_model_dir_TeamA} --policy_model_dir_teamB ${policy_model_dir_TeamB} --si_model_dir_teamA ${si_model_dir_TeamA} --si_model_dir_teamB ${si_model_dir_TeamB} \
--num_env_steps 50000000 --ppo_epoch 10 --clip_param 0.05 --use_value_active_masks --use_eval --eval_episodes 32 --eval_interval 20  --use_wandb  --minimal_gifs