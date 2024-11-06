import os

import matplotlib as mpl
import numpy as np
import torch
import wandb
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import get_shape_from_obs_space
from tensorboardX import SummaryWriter

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import math
from pathlib import Path
from types import SimpleNamespace

from group_social_influence.OmniSocialInfluence import (
    SocialInfluenceManager, convert_joint_to_individual)
from group_social_influence.smac_utility.translations_4m_v_5m_to_4m import *
import argparse


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)
class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.single_eval_envs = config['single_eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir
        self.no_train_policy = self.all_args.no_train_policy
        self.graph_node_positions = None
        # self.load_social_influence_model = self.all_args.load_social_influence_model
        self.model_dir_si_learning = self.all_args.model_dir_si_learning
        self.model_dir_si_fixed = self.all_args.model_dir_si_fixed
        self.evaluate_two_social_influence = self.all_args.evaluate_two_social_influence
        self.only_use_argmax_policy = self.all_args.only_use_argmax_policy
        self.only_use_argmax_action = self.all_args.only_use_argmax_action
        self.no_train_si = self.all_args.no_train_si
        self.minimal_gifs = self.all_args.minimal_gifs
        self.si_loss_type = self.all_args.si_loss_type
        assert self.si_loss_type in ["kl", "js", "bce"], "si_loss_type must be either kl, js, bce"
        
        self.run_mixed_population = self.all_args.run_mixed_population
        self.differentOriginEnvTeams = config['differentOriginEnvTeams']
        
                
        if self.run_mixed_population:
            print("Running mixed population team. 🏈 🏉")
            if self.differentOriginEnvTeams:
                print("Using different origin env teams. 💣 🧨")
                self.team_b_envs = config["teamB_envs"]
            else:
                print("Using same origin env teams. 🚀 🚀 Possibly different seeds?")
                
        self.train_si_on_joint_team_exp = self.all_args.train_si_on_joint_team_exp
        if self.no_train_si:
            print("Not training social influence. 🚫")
        else:
            if self.train_si_on_joint_team_exp:
                print("WARNING: Training SI on joint team experience 🟪. This should result in subpar compared to no train.")
            else: 
                print("Training social influence on individual team policy 🟥🟦. 🚀")
            
                
        self.graph_equality_type = self.all_args.graph_equality_type
        
        self.pruning_amount_for_group_si = self.all_args.pruning_amount_for_group_si

        
            
            
        ##### Try to make shorthand
        if self.all_args.env_name == "SMAC":
            from smac.env.starcraft2.maps import get_map_params
            num_agents = get_map_params(self.all_args.map_name)["n_agents"]
            num_enemies = get_map_params(self.all_args.map_name)["n_enemies"]
        elif self.all_args.env_name == 'StarCraft2':
            from onpolicy.envs.starcraft2.smac_maps import get_map_params
            num_agents = get_map_params(self.all_args.map_name)["n_agents"]
            num_enemies = get_map_params(self.all_args.map_name)["n_enemies"]
        elif self.all_args.env_name == "SMACv2" or all_args.env_name == 'StarCraft2v2':
            from smacv2.env.starcraft2.maps import get_map_params
            num_agents = parse_smacv2_distribution(self.all_args)['n_units']
            num_enemies = get_map_params(self.all_args.map_name)["n_enemies"]

        self.percent_team_A_agents = self.all_args.percent_team_A_agents
        SplittingPoint = math.floor(num_agents * (self.percent_team_A_agents /100))
        print("For gameplay, we have ", SplittingPoint, "agents in team A and ", num_agents - SplittingPoint, "agents in team B")
        
        
        
        if self.env_name == "3m":
            ShortHandActions = ["X", "H", "N", "S", "E", "W", "S1", "S2", "S3"]
        else:
            baseActions = ["X", "H", "N", "S", "E", "W"]
            ShootActions = ["S" + str(i+1) for i in range(num_enemies)]
            ShortHandActions = baseActions + ShootActions
            # if self.TwoDifferentTeamEnvs:
            #     ShootActions = ["S" + str(i+2) for i in range(num_enemies)] # 5m has extra enemy
            # ShortHandActions = baseActions + ShootActions
        self.ShortHandActions = ShortHandActions

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            # self.gifs_dir = str(wandb.run.dir + '/Gifs')
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.gifs_dir = str(config["run_dir"] / 'Gifs')
        if not os.path.exists(self.gifs_dir):
            os.makedirs(self.gifs_dir)
            
        # if self.evaluate_two_social_influence:
        #     first_gif_dir = str(config["run_dir"] / 'Gifs/DualSIComparison-Fixed')
        #     if not os.path.exists(first_gif_dir):
        #         os.makedirs(first_gif_dir)
                
        #     second_gif_dir = str(config["run_dir"] / 'Gifs/DualSIComparison-Learning')
        #     if not os.path.exists(second_gif_dir):
        #         os.makedirs(second_gif_dir)
        
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.algorithm.transformer_policy import \
                TransformerPolicy as Policy
            from onpolicy.algorithms.mat.mat_trainer import \
                MATTrainer as TrainAlgo
        else:
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import \
                R_MAPPOPolicy as Policy
            from onpolicy.algorithms.r_mappo.r_mappo import \
                R_MAPPO as TrainAlgo

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        share_observation_space_team_B = self.team_b_envs.share_observation_space[0] if self.use_centralized_V else self.team_b_envs.observation_space[0]
        
        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        print("num_agents: ", self.num_agents)
        
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], self.num_agents, device = self.device)
            
        else:
            if self.run_mixed_population:
                self.policy_A = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device) # This one for mappos
                action_space_teamB = self.envs.action_space[0]
                if self.differentOriginEnvTeams:
                    if isinstance(self.all_args, dict):
                        SpecialArgs = self.all_args.copy()
                    elif isinstance(self.all_args, argparse.Namespace):
                        SpecialArgs = vars(self.all_args).copy() # if wandb is off?
                    else:
                        SpecialArgs = dict(self.all_args) # if wandb is on?
                    SpecialArgs["map_name"] = config["teamB_map_name"]
                    ConfigStyleArgs = SimpleNamespace(**SpecialArgs)
                    ObservationSpace_teamB = self.team_b_envs.observation_space[0]
                    action_space_teamB = self.team_b_envs.action_space[0]
                    
                    print("Team B obs_space: ", ObservationSpace_teamB)
                    print("Team B share_obs_space: ", share_observation_space_team_B)
                    print("Team B act_space: ", action_space_teamB)
                    
                    self.policy_B = Policy(ConfigStyleArgs, ObservationSpace_teamB, share_observation_space_team_B, action_space_teamB, device = self.device) # This one for mappo
            else:
                self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device = self.device) # This one for mappo
                
        if self.run_mixed_population:
            # assert self.all_args.policy_model_dir_teamA is not None and self.all_args.policy_model_dir_teamB is not None, "You need to provide two policy models for mixed population"
            assert self.all_args.policy_model_dir_teamA is not None, "You need to provide at least one policy models for mixed population"
            self.policy_model_dir_teamA = self.all_args.policy_model_dir_teamA
            self.policy_model_dir_teamB = self.all_args.policy_model_dir_teamB

        if self.model_dir is not None or self.run_mixed_population:
            if self.run_mixed_population:
                # TODO restore two models
                self.restore(self.policy_model_dir_teamA, self.policy_A)
                if self.policy_model_dir_teamB is not None:
                    self.restore(self.policy_model_dir_teamB, self.policy_B)

            else:
                self.restore(self.model_dir, self.policy)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            if self.run_mixed_population:
                self.trainer_A = TrainAlgo(self.all_args, self.policy_A, device = self.device) # This one for mappo
                self.trainer_B = TrainAlgo(self.all_args, self.policy_B, device = self.device) # This one for mappo
            else:
                self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device) # This one for mappo
            
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0], run_mixed_population=self.run_mixed_population, differentOriginEnvTeams=self.differentOriginEnvTeams)
        
        self.buffer_B = None
        if self.run_mixed_population and self.differentOriginEnvTeams:
            self.buffer_B = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        ObservationSpace_teamB,
                                        share_observation_space_team_B,
                                        action_space_teamB, run_mixed_population=self.run_mixed_population, differentOriginEnvTeams=self.differentOriginEnvTeams)
            print("We have two buffers, since we have differentOriginEnvTeams.")
 
        
        self.eval_buffer = SharedReplayBuffer(self.all_args,
                                    self.num_agents,
                                    self.envs.observation_space[0],
                                    share_observation_space_team_B,
                                    self.envs.action_space[0], evaluation=True, run_mixed_population=self.run_mixed_population)


        action_space = self.envs.action_space[0]
        
        
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.mujoco_box = True
            action_dim = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
        else:  # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
                        
        # Social influence has addition of action to all things
        n_obs = get_shape_from_obs_space(self.envs.observation_space[0])[0]
        # n_shared_obs = get_shape_from_obs_space(self.envs.share_observation_space[0])[0]
        social_influence_observation_space = n_obs + action_dim
        social_influence_prediction_space = action_dim * (self.num_agents -1)
        
        print("social_influence_net_policy obs_space: ", social_influence_observation_space)
        # print("social_influence_net_policy share_obs_space: ", social_influence_share_observation_space)
        print("social_influence_net_policy prediction_space: ", social_influence_prediction_space)
        
        params = {"num_agents": self.num_agents, 
                  "action_space": action_dim, 
                  "action_dim": 1, # because discrete action for SMAC
                  "discrete_actions": True,   
                  "n_obs": n_obs, 
                  "n_hidden_units": self.hidden_size, 
                  "device": self.device,    
                  "social_influence_n_counterfactuals": 20, # this is a param set below this is a placeholder value
                  "num_env_steps": self.num_env_steps,
                  }
        
        # social_influence_n_counterfactuals = 50 # if continous
        if params["discrete_actions"]:            
            social_influence_n_counterfactuals = int(action_dim) - 2 # because removing the agent's own action and the noops action 0
            
        if self.run_mixed_population:
            custom_teamB_params = {}
            team_B_action_space = self.team_b_envs.action_space[0]
            if self.differentOriginEnvTeams:
                if team_B_action_space.__class__.__name__ == "Discrete":
                    team_B_action_dim = team_B_action_space.n
                elif team_B_action_space.__class__.__name__ == "Box":
                    self.mujoco_box = True
                    team_B_action_dim = team_B_action_space.shape[0]
                elif team_B_action_space.__class__.__name__ == "MultiBinary":
                    team_B_action_dim = team_B_action_space.shape[0]
                else: 
                    print("Error with team b action space")

                custom_teamB_params =  {"num_agents": self.num_agents, 
                    "action_space": team_B_action_dim, 
                    # "action_dim": 1, # because discrete action for SMAC
                    "discrete_actions": True,   
                    "n_obs": get_shape_from_obs_space(self.team_b_envs.observation_space[0])[0], 
                    # "n_hidden_units": self.hidden_size, 
                    # "device": self.device,    
                    # "num_env_steps": self.num_env_steps,
                    # "gifs_dir": self.gifs_dir,
                    # "social_influence_n_counterfactuals": self.social_influence_n_counterfactuals, # We don't want to compute for non-existent action here
                    # "si_loss_type": self.si_loss_type,
                    # "only_use_argmax_policy": self.only_use_argmax_policy,
                    }
                
            # change SI directory input 
            # assert self.all_args.si_model_dir_teamA is not None and self.all_args.si_model_dir_teamB is not None, "You need to provide two si models for mixed population evaluation"
            assert self.all_args.si_model_dir_teamA is not None, "You need to provide at least 1 si models for mixed population evaluation"
            self.si_model_dir_teamA = self.all_args.si_model_dir_teamA
            self.si_model_dir_teamB = self.all_args.si_model_dir_teamB
            self.social_influence_network = SocialInfluenceManager(num_agents=self.num_agents, hidden_size=256, device=self.device, num_env_steps=self.num_env_steps, action_space=action_dim, n_obs=n_obs, social_influence_n_counterfactuals=social_influence_n_counterfactuals, discrete_actions=params["discrete_actions"], gifs_dir=self.gifs_dir, model_dir_si_fixed=self.si_model_dir_teamA, model_dir_si_learning= self.si_model_dir_teamB, compare_si_policy=False, run_mixed_population= self.run_mixed_population, si_loss_type=self.si_loss_type, only_use_argmax_policy=self.only_use_argmax_policy, ShortHandActions=self.ShortHandActions, TwoDifferentTeamEnvs=self.differentOriginEnvTeams, TeamBParams=custom_teamB_params)
            assert self.evaluate_two_social_influence == False, "You are doing mixed population so no evaluating two different social influence networks"
            
            print("Team A action space is: ", action_dim) #10
            print("Team B action space is: ", team_B_action_dim) #11
        else:
            self.social_influence_network = SocialInfluenceManager(num_agents=self.num_agents, hidden_size=256, device=self.device, num_env_steps=self.num_env_steps, action_space=action_dim, n_obs=n_obs, social_influence_n_counterfactuals=social_influence_n_counterfactuals, discrete_actions=params["discrete_actions"], gifs_dir=self.gifs_dir, model_dir_si_fixed=self.model_dir_si_fixed, model_dir_si_learning= self.model_dir_si_learning, compare_si_policy=self.evaluate_two_social_influence, si_loss_type=self.si_loss_type, only_use_argmax_policy=self.only_use_argmax_policy, ShortHandActions=self.ShortHandActions)
        
        # self.social_influence_net = SocialInfluenceTrainer(self.policy, params) 
        
        

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, group_social_influence, group_social_influence_factor=0.5):
        """Calculate returns for the collected data."""
        if self.run_mixed_population:
            self.trainer_A.prep_rollout()
            self.trainer_B.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(self.buffer.obs[-1]),
                                                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]))
                next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
                self.buffer.social_compute_returns(next_values, self.trainer_A.value_normalizer, group_social_influence=group_social_influence, group_social_influence_factor=group_social_influence_factor) # Can just use A because this is the same for both
            else:
                if self.differentOriginEnvTeams:
                    rnnStatesCritic = self.buffer.rnn_states_critic
                else: 
                    rnnStatesCritic = self.buffer.rnn_states_critic_A
                next_values = self.trainer_A.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(rnnStatesCritic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]))
                next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
                self.buffer.social_compute_returns(next_values, self.trainer_A.value_normalizer, group_social_influence=group_social_influence, group_social_influence_factor=group_social_influence_factor)     
                
                # Because either way need to fix up for trainer b
                if self.differentOriginEnvTeams:
                    BufferToUse = self.buffer_B
                    rnnStatesCritic = self.buffer_B.rnn_states_critic
                else:
                    BufferToUse = self.buffer
                    rnnStatesCritic = self.buffer.rnn_states_critic_B
                
                next_values = self.trainer_B.policy.get_values(np.concatenate(BufferToUse.share_obs[-1]),
                                                            np.concatenate(rnnStatesCritic[-1]),
                                                            np.concatenate(BufferToUse.masks[-1]))
                
                
                next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            
                BufferToUse.social_compute_returns(next_values, self.trainer_B.value_normalizer, group_social_influence=group_social_influence, group_social_influence_factor=group_social_influence_factor)     
            
            
        else:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(self.buffer.obs[-1]),
                                                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]))
            else:
                next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                            np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                            np.concatenate(self.buffer.masks[-1]))
            next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
            
            
            
            
            
            self.buffer.social_compute_returns(next_values, self.trainer.value_normalizer, group_social_influence=group_social_influence, group_social_influence_factor=group_social_influence_factor)
            
    # def single_timestep_eval_compute_graph_similarity_group_social_influence_rewards(self, obsA, actionsA, obsB):
    #     IndividualObs_A, actions_prepped, _ = self.prep_data(Obs_A, actions, rewards=None, true_policies=None, time_adjust=False)
    #     IndividualObs_B, _, _ = self.prep_data(Obs_B, actions, rewards=None, true_policies=None, time_adjust=False)


    def eval_compute_group_social_influence_rewards(self, Obs_A, actions, Obs_B, AlreadyProcessed=False, episode=0):
        if self.run_mixed_population:
            if AlreadyProcessed:
                # IndividualObs_A = convert_joint_to_individual(Obs_A)
                # IndividualObs_B = convert_joint_to_individual(Obs_B)
                IndividualObs_A = Obs_A
                IndividualObs_B = Obs_B
                # actions_prepped = convert_joint_to_individual(actions)
                actions_prepped = actions
            else:
                if self.differentOriginEnvTeams:
                    Obs_B = allThingsTranslationPadding(Obs_A, TypeDeclaration="observation")
                else: 
                    Obs_B = Obs_A

                IndividualObs_A, actions_prepped, _ = self.prep_data(Obs_A, actions, rewards=None, true_policies=None, time_adjust=False)
                IndividualObs_B, _, _ = self.prep_data(Obs_B, actions, rewards=None, true_policies=None, time_adjust=False)
            # print("Input sizes for social influence rewards: ", Obs_A.shape, Obs_B.shape, actions.shape)
            # print("Sizes of IndividualObs_A, IndividualObs_B, actions_prepped: ", IndividualObs_A.shape, IndividualObs_B.shape, actions_prepped.shape)

            # this is the graph similarity reward
            # group_social_influence_rewards = self.social_influence_network.compute_group_social_influence_rewards(IndividualObs_A, actions_prepped, TeamBObs = IndividualObs_B)
            group_social_influence_rewards = self.social_influence_network.more_efficient_compute_group_social_influence_rewards_all_types(IndividualObs_A, actions_prepped, TeamBObs = IndividualObs_B, chosen_pruning_percentile=self.pruning_amount_for_group_si, episode=episode)
            
  

            return group_social_influence_rewards

        else:
            IndividualObs, IndividualActions, _ = self.prep_data(Obs_A, actions, rewards=None, true_policies=None, time_adjust=False)
            ## Calculate the social influence rewards 
            perAgentSocialInfluenceRewards = self.social_influence_network.compute_group_social_influence_rewards(IndividualObs, IndividualActions)

            metrics = {"Social Influence Rewards": perAgentSocialInfluenceRewards}
    
            # self.buffer.set_group_social_influence_rewards(perAgentSocialInfluenceRewards)

            return perAgentSocialInfluenceRewards
        
        return metrics
        
    def compute_group_social_influence_rewards(self, episode=0):
        """Calculate group social influence rewards for the collected data."""
        # TODO Fix to properly train social influence
        
        if self.run_mixed_population:
            self.trainer_A.prep_rollout()
            self.trainer_B.prep_rollout()
        else:
            self.trainer.prep_rollout()
            
        if self.run_mixed_population and self.differentOriginEnvTeams:
            BufferToUse = [self.buffer, self.buffer_B]
            IndividualObs_A, actions, _ = self.buffer_prep_data(time_adjust=False, WhichBufferToUse=self.buffer)
            IndividualObs_B, _, _ = self.buffer_prep_data(time_adjust=False, WhichBufferToUse=self.buffer_B)
            # this is the graph similarity reward
            group_social_influence_rewards = self.social_influence_network.compute_group_social_influence_rewards(IndividualObs_A, actions, TeamBObs = IndividualObs_B, chosenRewardType=self.graph_equality_type, chosen_pruning_percentile=self.pruning_amount_for_group_si)

            self.buffer.set_group_social_influence_rewards(group_social_influence_rewards)
            self.buffer_B.set_group_social_influence_rewards(group_social_influence_rewards)
            
            # print("Set group social influence rewards: Sum is ", np.sum(group_social_influence_rewards))

            return group_social_influence_rewards

        else:
            BufferToUse = self.buffer
            IndividualObs, IndividualActions, _ = self.buffer_prep_data(time_adjust=False, WhichBufferToUse=BufferToUse)
            ## Calculate the social influence rewards 
            perAgentSocialInfluenceRewards = self.social_influence_network.compute_group_social_influence_rewards(IndividualObs, IndividualActions)

            # metrics = {"Social Influence Rewards": perAgentSocialInfluenceRewards}
    
            self.buffer.set_group_social_influence_rewards(perAgentSocialInfluenceRewards)

            return perAgentSocialInfluenceRewards

        return None
        
    # def compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(self, obs, actions, reward, total_num_steps=0, positionOfAgents=None, minimal_gifs=False, frames=None):
        
    #     obs, actions, _ = self.prep_data(obs, actions, reward, None, time_adjust=False)
        
        
    #     metrics = self.social_influence_network.compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(obs, actions, total_num_steps, positionOfAgents, minimal_gifs=minimal_gifs, frames=frames)

    #     return metrics
    
    def process_from_teamA_to_teamB(self, obs, actions, policies): 
        """TODO check out how SMAC does this - what is the size I need to reshape to etc. Also do this for the state? Another function?"""
        
        obs = allThingsTranslationPadding(obs, TypeDeclaration="observation")
        policies = allThingsTranslationPadding(policies, TypeDeclaration="policy")
        
        # No need to do the actions as these should not be one hot encoded, but just the action number, so same size

        return obs, actions, policies
    
    def compute_group_social_influence_rewards_evaluation_single_episode(self, obs, actions, reward, total_num_steps=0, positionOfAgents=None, minimal_gifs=False, frames=None, policies = None):
        obs, actions, _ = self.prep_data(obs, actions, reward, None, time_adjust=False)
        
        if policies is not None:
            policies = self.prep_policies_for_visualization(policies)
            
        if self.run_mixed_population:            
            metricsA = self.social_influence_network.compute_group_social_influence_rewards_evaluation_single_episode(obs, actions, total_num_steps, positionOfAgents, minimal_gifs=minimal_gifs, frames=frames, policies = policies, whichNetToUse=self.social_influence_network.social_influence_net_A, pruning_amount_for_group_similarity=self.pruning_amount_for_group_si)
            
            if self.differentOriginEnvTeams:
                obsB, actionsB, policiesB = self.process_from_teamA_to_teamB(obs, actions, policies) #TODO Pam! Fixing diff env
            else: 
                obsB, actionsB, policiesB = obs, actions, policies
            
            metricsB = self.social_influence_network.compute_group_social_influence_rewards_evaluation_single_episode(obsB, actionsB, total_num_steps, positionOfAgents, minimal_gifs=minimal_gifs, frames=frames, policies = policiesB, whichNetToUse=self.social_influence_network.social_influence_net_B, pruning_amount_for_group_similarity=self.pruning_amount_for_group_si)
            
            metricsA = {k + "_(Team A)": v for k, v in metricsA.items()}
            metricsB = {k + "_(Team B)": v for k, v in metricsB.items()}

            metrics = {**metricsA, **metricsB}

            groupSimilarityMetrics = self.eval_compute_group_social_influence_rewards(obs, actions, obsB, AlreadyProcessed=True, episode=total_num_steps)
            
            # add on 
            groupSimilarityMetrics = {"single_ep_eval/" + k: v for k, v in groupSimilarityMetrics.items()}
            
            metrics.update(groupSimilarityMetrics)

            # make a graph here (SI at both sides and game in the middle, graph similarity scores)

            comboMetrics = self.social_influence_network.compute_dual_SI_group_social_influence_diagram_evaluation_single_episode(obs, actions, obsB, positionOfAgents=positionOfAgents, frames=frames, policiesA=policies, policiesB=policiesB, total_num_steps=total_num_steps) # note these are the same policies just padded and not because these are the policies that agents actually use

            metrics.update(comboMetrics) # adds graph to metrics
            

        else:
                
            metrics = self.social_influence_network.compute_group_social_influence_rewards_evaluation_single_episode(obs, actions, total_num_steps, positionOfAgents, minimal_gifs=minimal_gifs, frames=frames, policies = policies, pruning_amount_for_group_similarity=self.pruning_amount_for_group_si)
        
        return metrics

    
    def train(self):
        """Train policies with data in buffer. """
        if self.run_mixed_population:
            self.trainer_A.prep_training()
            self.trainer_B.prep_training()
            train_infos_A = self.trainer_A.train(self.buffer)
            # Append 'A' to the end of the key
            train_infos_A = {k + "_(Team A)": v for k, v in train_infos_A.items()}
            
            if self.differentOriginEnvTeams:
                BufferToUse = self.buffer_B
            else: 
                BufferToUse = self.buffer
            train_infos_B = self.trainer_B.train(BufferToUse)
            train_infos_B = {k + "_(Team B)": v for k, v in train_infos_B.items()}
            
            
            train_infos = {**train_infos_A, **train_infos_B}
            
        else:
            self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer)      
            
        return train_infos
    
    def buffer_prep_data(self, time_adjust, WhichBufferToUse=None):
        if WhichBufferToUse is None:
            WhichBufferToUse = self.buffer
        joint_obs = WhichBufferToUse.obs
        actions = WhichBufferToUse.actions
        true_policies = WhichBufferToUse.policies
        rewards = WhichBufferToUse.rewards
        
        return self.prep_data(joint_obs, actions, rewards, true_policies, time_adjust)
    
    def prep_policies_for_visualization(self, policies):
        #(18, 1, 3, 9) -> (3, 18, 9)
        # first remove the 1
        policies = policies.squeeze(1) # (18, 3, 9)
        # I want 3 18 9
        policies = policies.transpose(1, 0, 2)

        return policies
    
    def prep_data(self, OG_joint_obs, OG_actions, rewards = None, true_policies = None, time_adjust = True):
        """time adjust = True is for when you need to remove the last obs and action, and also need the true next policy for training. for loss calculation.
        Have it false when you are just calculating reward
        
        """
        
        if true_policies is not None:
            # Assuming true_policies is your 4D numpy array
            # Remove the first element of each episode
            next_policies = true_policies[1:, :, :, :] # First policy is never used
            
        if time_adjust: # for calculating losses where you need true policy later, but also no need last obs and action
            joint_obs = OG_joint_obs[:-1, :, :, :][:-1]
            actions = OG_actions[:-1, :, :, :]
        else:
            # no need to adjust time for training
            joint_obs = OG_joint_obs[:-1] # SMAC quirk
            actions = OG_actions
            # return joint_obs, actions, None

        #### Reshaping
        ListOfTruePoliciesOfAgents = None
        if len(OG_joint_obs.shape) == 4:
            # keep (self.episode_length, num_agent, dim) - getting rid of 8 roll out threads from second pos
            if rewards is not None:
                episode_length, n_rollout_threads, num_agents = rewards.shape[0:3]
            else: 
                episode_length, n_rollout_threads, num_agents = joint_obs.shape[0:3]
            accounted_for_batch_size = n_rollout_threads * (episode_length - 1)  # because you have cut off the last experience
            batch_size = n_rollout_threads * episode_length
            if time_adjust:
                batch_size = accounted_for_batch_size
                
            joint_obs = joint_obs.reshape((batch_size, *OG_joint_obs.shape[2:])) # obs have an extra, so remove the last one
            actions = actions.reshape((batch_size, *OG_actions.shape[2:]))
            if true_policies is not None:
                ListOfTruePoliciesOfAgents = next_policies.reshape((accounted_for_batch_size, *true_policies.shape[2:])) # always cutting off the last experience
            
        # Make this list of joint obs into individual obs list, e.g. switch from [timestep, agent, obs] to [agent, timestep, obs]  
        IndividualObs = convert_joint_to_individual(joint_obs)
        JointActions = convert_joint_to_individual(actions)
        IndividualActions = JointActions.float()
            
        return IndividualObs, IndividualActions, ListOfTruePoliciesOfAgents
    
    
    def get_loss(self):
        """Calc loss with data in buffer. no training"""
        if self.run_mixed_population:
            # self.trainer_A.prep_training()
            # self.trainer_B.prep_training()

            IndividualObs_A, IndividualActions_A, next_policies_A = self.buffer_prep_data(time_adjust=True, WhichBufferToUse=self.buffer)


            train_infos_A = self.social_influence_network.get_loss(IndividualObs_A, IndividualActions_A, next_policies_A, AGENTS_CAN_DIE = True)
            # Append 'A' to the end of the key
            train_infos_A = {k + "_(Team A)": v for k, v in train_infos_A.items()}

            train_infos = train_infos_A # abstracted to the omnisocialinfluence side
            
            # if self.differentOriginEnvTeams:
            #     BufferToUse = self.buffer_B
            # else: 
            #     BufferToUse = self.buffer

            # IndividualObs_B, IndividualActions_B, next_policies_B = self.buffer_prep_data(time_adjust=True, WhichBufferToUse=BufferToUse)

            # train_infos_B = self.social_influence_network.get_loss(IndividualObs_B, IndividualActions_B, next_policies_B, AGENTS_CAN_DIE = True)

            # train_infos_B = {k + "_(Team B)": v for k, v in train_infos_B.items()}
            
            
            # train_infos = {**train_infos_A, **train_infos_B}
        else:
            IndividualObs, IndividualActions, next_policies = self.buffer_prep_data(time_adjust=True)
            
            train_infos = self.social_influence_network.get_loss(IndividualObs, IndividualActions, next_policies, AGENTS_CAN_DIE = True)


        return train_infos
    
    def train_social_influence(self):
        """Train policies with data in buffer. """
        
        joint_obs, actions, true_policies = self.buffer_prep_data(time_adjust=True)

        train_infos = self.social_influence_network.train(joint_obs, actions, true_policies, AGENTS_CAN_DIE=False)

        return train_infos
    
    def clear_gifs(self):
        self.social_influence_network.clear_gifs()
    
    def finished_training(self):
        self.buffer.after_update()
        if self.run_mixed_population and self.differentOriginEnvTeams:
            self.buffer_B.after_update()
        
    def _save(self, episode=0, episodeLabel=False, prefix="", whichTrainer=None):
        """Save policy's actor and critic networks."""
        saveDir = str(self.save_dir) + prefix
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(saveDir, episode)
        else:
            policy_actor = whichTrainer.policy.actor
            torch.save(policy_actor.state_dict(), str(saveDir) + "/actor.pt")
            policy_critic = whichTrainer.policy.critic
            torch.save(policy_critic.state_dict(), str(saveDir) + "/critic.pt")
            
            if episodeLabel: # because eval interval
                NumSteps = f"{episode/1000000:.1f}M"
                EpisodeSaveDir = str(saveDir) + "/Policy_Episode" + str(NumSteps)

                Path(EpisodeSaveDir).mkdir(parents=True, exist_ok=True)

                policy_actor = whichTrainer.policy.actor
                torch.save(policy_actor.state_dict(), str(EpisodeSaveDir) + "/actor.pt")
                policy_critic = whichTrainer.policy.critic
                torch.save(policy_critic.state_dict(), str(EpisodeSaveDir) + "/critic.pt")

    def save(self, episode=0, episodeLabel=False):
        """Save policy's actor and critic networks."""
        if self.run_mixed_population:
            self._save(episode, episodeLabel, prefix="TeamA", whichTrainer=self.trainer_A)
            self._save(episode, episodeLabel, prefix="TeamB", whichTrainer=self.trainer_B)
        else:
            self._save(episode, episodeLabel, prefix="", whichTrainer=self.trainer)
            
    def save_si(self, episode=0, episodeLabel=False):
        """Save si networks."""
        self.social_influence_network.save_si(self.save_dir, episode, episodeLabel)
        

    def restore(self, model_dir, policyTarget = None):
        """Restore policy's networks from a saved model."""
        if policyTarget is None:
            policyTarget = self.policy
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            policyTarget.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(os.path.join(model_dir, 'actor.pt'))
            policyTarget.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render and self.all_args.load_critic:
                policy_critic_state_dict = torch.load(os.path.join(model_dir, 'critic.pt'))
                policyTarget.critic.load_state_dict(policy_critic_state_dict)
                
        #social influence
    def restore_social_influence(self, model, model_dir):
        model.load(model_dir)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.summary_writer.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.summary_writer.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    
