# PPO code to play with the toy environment
# Path: CooperationMetricChecks/MultiOmniAgentSkillTester.py
import argparse
import copy
import logging
import os
import random
import time
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import wandb
from smac.env import StarCraft2Env

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

from OmniSocialInfluence import GroupSocialInfluence



def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Add the arguments

    parser.add_argument('--total_episodes', type=int, default=10000, help='Total number of episodes to run. Default is 10000')
    
    parser.add_argument('--notes', type=str, default="", help='Anything Extra to doo')
    
    parser.add_argument("--n_agents", type=int, default=2, help="Number of agents in the environment")
    parser.add_argument("--run_id", type=str, default="tester", help="Run ID for the experiment")
    
    # bool argument turn on when it is used
    parser.add_argument("--social_influence", action="store_true", help="Whether or not to use social influence reward, will compute it either way so we can compare the two.")
    parser.add_argument("--social_influence_n_counterfactuals", type=int, default=10, help="Number of counterfactuals to use for social influence")
    
    parser.add_argument("--n_hidden_units", type=int, default=256, help="Number of hidden units in the policy network, value network, skill network and social influence network.")
    
    # Parse the arguments
    args = parser.parse_args()
    # Convert to dictionary
    args = vars(args)

    params = {
        "size_of_binary_encoding": args["n_skills"].bit_length(),
        "total_training_steps": args["total_episodes"], # *1000, # Because T is 1000
    }

    params.update(args)
    logging.info(params)
    
    ## Logic check 
    MapDimensions = 2
    if params["single_corridor_mode"]:
        MapDimensions = 1
    if params["social_influence_scale"] != 10.0  and not params["social_influence"]:
        logging.warning("Social influence scale is set but social influence is not being used.")
        
    assert params["skill_reward_type"] in ["next_state", "change_in_state"], "Skill reward type must be next_state or change_in_state"
    assert params["diamond_count"] <= params["map_size"] ** MapDimensions, "Diamond count must be less than or equal to map spaces."
    if "diamond" in params["filtered_skill_observation"]:
        assert params["diamond_count"] > 0, "Diamond count must be greater than 0 for diamond observed in filtered skill state"
    else: 
        assert params["diamond_count"] == 0, "Diamond count must be 0 for diamond not observed in filtered skill state"
        
    assert params["n_agents"] <= params["map_size"] ** MapDimensions, "Agent count must be less than or equal to map size spaces"
    
    return params

def initialize_wandb(params):
    run_name = "CompareMetric" + "_" + params["env_name"] + "_" + params["run_id"] 
    # truncate run name if it is too long
    if len(run_name) > 100:
        run_name = run_name[:100]
    
    #"_n_skills_" + str(params["n_skills"]) + "_SkillRewardType_" + params["skill_reward_type"]
    # Add date time to the run name - more readable (YYYY-MM-DD-HH-MM-SS)
    run_name += datetime.now().strftime("_%y-%m-%d (%H-%M)")
    
    
    # USE_WANDB = True
    if params["use_wandb"]:
        wandb.login(key="83385f5ec22fac4705b677a03b63a470e70d61da")
        wandb.init(
            save_code=True,
            tags=["MultiPointerEnv", "MultiAgentSkills"],
            project="Multi Agent Joint Skills",
            name=run_name,
            entity="moms-2023",
            config=params,
            notes=params["notes"],
            resume="allow",
            allow_val_change=True,
        )
    return run_name



def initialize_environment_and_policy(params):
    # Initialize environment and policy
    # MyEnv = ToyEnvOmniMulti(max_map_size=params["map_size"], max_step=100, filtered_skill_observation=params["filtered_skill_observation"],other_agent_observation=params["other_agent_observation"], agent_count=params["n_agents"], diamond_count = params["diamond_count"], single_corridor_mode= params["single_corridor_mode"])
    
    MyEnv = StarCraft2Env(map_name=params["env_name"])
    
    params["action_space"] = MyEnv.individual_action_space.n
    params["action_dim"] = MyEnv.individual_action_size # 1 # For my discrete environment
    params["n_obs"] = MyEnv.individual_observation_space.shape[0]
    params["discrete_actions"] = True
    
    # Initialize policy with PPO hyperparameters
    policy = PolicyNetwork(MyEnv, skill_size=params["size_of_binary_encoding"])
    value_network = ValueNetwork(MyEnv, skill_size=params["size_of_binary_encoding"])
    skill_network = SkillNetwork(MyEnv, skill_size=params["n_skills"]) # One hot encoding of skills
    
    old_policy_network = copy.deepcopy(policy)
    
    return MyEnv, policy, old_policy_network, value_network, skill_network, params

def initialize_for_social_influence(policy_network, params):
    
    group_social_influence_network = GroupSocialInfluence(policy_network, params)
    
    learning_rate_si = 3e-4
    weight_decay_si = 1e-5
    optimizer_epsilon_si = 1e-5 # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    
    optimizers_si_list = []
    scheduler_si_list = []
    for social_influence_network in group_social_influence_network.social_influence_networks:
        optimizer_si = optim.Adam(social_influence_network.parameters(), lr=learning_rate_si, weight_decay=weight_decay_si, eps=optimizer_epsilon_si)
        scheduler_si = torch.optim.lr_scheduler.LambdaLR(optimizer_si, lr_lambda=lambda step: 1 - step / params["total_training_steps"])
        optimizers_si_list.append(optimizer_si)
        scheduler_si_list.append(scheduler_si)
        
    group_social_influence_network.optimizers_list = optimizers_si_list
    group_social_influence_network.schedulers_list = scheduler_si_list

    return group_social_influence_network, optimizers_si_list, scheduler_si_list

def initialize_optimizers(policy, value_network, skill_network, params):
    # Initialize the optimizer outside the function
    learning_rate_policy = 1e-4
    learning_rate_value = 1e-3
    # learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer_epsilon = 1e-5 # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    optimizer_skill = optim.Adam(skill_network.parameters(), lr=learning_rate_policy, weight_decay=weight_decay, eps=optimizer_epsilon)
    optimizer_policy = optim.Adam(policy.parameters(), lr=learning_rate_policy, weight_decay=weight_decay, eps=optimizer_epsilon)
    optimizer_value = optim.Adam(value_network.parameters(), lr=learning_rate_value, weight_decay=weight_decay, eps=optimizer_epsilon)
    
    # Define a lambda function for the learning rate decay
    lr_lambda = lambda step: 1 - step / params["total_training_steps"]
    # Create a learning rate scheduler for each optimizer
    scheduler_skill = torch.optim.lr_scheduler.LambdaLR(optimizer_skill, lr_lambda)
    scheduler_policy = torch.optim.lr_scheduler.LambdaLR(optimizer_policy, lr_lambda)
    scheduler_value = torch.optim.lr_scheduler.LambdaLR(optimizer_value, lr_lambda)
    
    return optimizer_skill, optimizer_policy, optimizer_value, scheduler_skill, scheduler_policy, scheduler_value


def main(params):
    run_name = initialize_wandb(params)
    MyEnv, policy, old_policy_network, value_network, skill_network, params = initialize_environment_and_policy(params)
    optimizer_skill, optimizer_policy, optimizer_value, scheduler_skill, scheduler_policy, scheduler_value = initialize_optimizers(policy, value_network, skill_network, params)
    
    group_social_influence_network, optimizers_si, scheduler_si = initialize_for_social_influence(policy, params)
    
    GifsDirectory = os.path.join("Gifs/Simple_Pointer_Env", run_name)
    os.makedirs(GifsDirectory, exist_ok=True)
    logging.info(f"Saving gifs at {GifsDirectory}")

    run_name = initialize_wandb(params)

    # Print difficulty analysis of the environment - compute_baseline_skill_reward
    # lower_bound, upper_bound, input_to_associate_per_skill = compute_baseline_skill_reward(MyEnv.individual_action_space.n, params)
    
    
    all_optimizers = [optimizer_skill, optimizer_policy, optimizer_value, scheduler_skill, scheduler_policy, scheduler_value]

    AllSkillRewards = []
    FrequencyToTest = params["total_episodes"] / 1000
    for episode in tqdm(range(params["total_episodes"])):
        myInfo = {}
        is_hundredth_episode = episode % 100 == 0
        is_thousandth_episode = episode % 1000 == 0
        if is_thousandth_episode:
            logging.info(f"Episode {episode}")
            start_time = time.time()
        # Collect data and update policy - ROLL OUT PHASE
        states, filtered_skill_states, joint_obs, actions, rewards, values, skills, env_rewards = collect_data(MyEnv, policy, value_network, skill_network, episode, params, T=500)
        
        training_reward = rewards # Not using env_rewards
        # Social influence data 
        if params["social_influence"]:
            si_reward, graph = group_social_influence_network.calc_social_influence_reward_group(joint_obs, actions)
            # training_reward += weighted_si_reward
        else: 
            graph = None
        
        if is_thousandth_episode:
            logging.info(f"Time for data collection: {time.time() - start_time}")
            start_time = time.time()
        loss_info, reporting_info = update_policy(all_optimizers, old_policy_network, policy, value_network, states, joint_obs, actions, training_reward, graph, values, skills, params) # env_rewards not used
        if is_thousandth_episode:
            logging.info(f"Time for policy update: {time.time() - start_time}")
            start_time = time.time()
        loss_info_skills = update_skill_network(optimizer_skill = optimizer_skill, scheduler_skill = scheduler_skill, skill_network=skill_network, filtered_states=filtered_skill_states, skills=skills, params=params)
        if is_thousandth_episode:
            logging.info(f"Time for skill network update: {time.time() - start_time}")
            start_time = time.time()
        
        # Social influence
        loss_info_si = group_social_influence_network.train_self(joint_obs, skills, actions)
            
        if is_thousandth_episode:
            logging.info(f"Time for social influence network update: {time.time() - start_time}")
            start_time = time.time()
        # At the end of each training iteration, update the old policy network
        old_policy_network.load_state_dict(policy.state_dict())
            
        myInfo.update(loss_info)
        myInfo.update(loss_info_skills)
        myInfo.update(loss_info_si)
        myInfo.update(reporting_info)
        
        # Test the environment every FrequencyToTest episodes
        if episode % FrequencyToTest == 0:
            with torch.no_grad():
                test_info, past_mean_skill_reward = compare_skills(MyEnv, policy, skill_network, group_social_influence_network, episode, params, past_mean_skill_reward, GifsDirectory)
                SkillRewards = [test_info[f"MultiOmniAgent-Test/Skill {skill} DIAYN Reward (Ep Mean)"] for skill in range(params["n_skills"])]
                AllSkillRewards.append(SkillRewards)
                #####
                if is_thousandth_episode:
                    logging.info(f"Time for comparing skills: {time.time() - start_time}")
                    start_time = time.time()
                myInfo.update(test_info)
                    
        # Log the losses
        wandb.log(myInfo, step=episode)
        if is_thousandth_episode:
            logging.info(f"Time for logging: {time.time() - start_time}")
    # Save Model Weights
    # Directory to save model weights
    ChkptDirectory = os.path.join("Checkpoints/Simple_Pointer_Env", run_name)
    os.makedirs(ChkptDirectory, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(ChkptDirectory, "policy.pth"))
    torch.save(value_network.state_dict(), os.path.join(ChkptDirectory, "value_network.pth"))
    torch.save(skill_network.state_dict(), os.path.join(ChkptDirectory, "skill_network.pth"))
            
    
    
if __name__ == "__main__":
    params = parse_arguments()
    main(params)
    
