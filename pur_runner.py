import time
import wandb
import numpy as np
from functools import reduce
import torch
from base_runner import Runner
import imageio
import numpy as np
import traceback
import math
### Trying to get pygame to render the environment
# import os
# os.environ["SDL_VIDEODRIVER"] = "directfb"


def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        start_time = time.time()
    
        # print("Starting warmup...")
        # warmup_start = time.time()
        self.warmup()   
        # warmup_end = time.time()
        # print(f"Warmup completed in {warmup_end - warmup_start:.2f} seconds")

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)
        
        

        for episode in range(episodes):
            print(f"##### Episode: {episode}")
            if self.use_linear_lr_decay:
                if self.run_mixed_population:
                    self.trainer_A.policy.lr_decay(episode, episodes)
                    self.trainer_B.policy.lr_decay(episode, episodes)
                else:
                    self.trainer.policy.lr_decay(episode, episodes)

            TotalCollectTime = 0 
            TotalInsertTime = 0

            episode_start = time.time()
            for step in range(self.episode_length):
                # Sample actions
                collect_start = time.time()
                if self.run_mixed_population:
                    values, actions, action_log_probs, rnn_states_A, rnn_states_B, rnn_states_critic_A, rnn_states_critic_B, policies = self.mixed_team_collect(step)

                else:
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = self.collect(step)
                collect_end = time.time()
                TotalCollectTime += collect_end - collect_start
                    
                # Obser reward and next obs
                env_actions = np.reshape(actions,(-1))
                # observations, rewards, terminations, truncations, infos
                obs, rewards, dones, trunks, infos = self.envs.step(env_actions)
                obs = np.reshape(obs,self.buffer.obs[0].shape)
                share_obs = obs.copy()
                available_actions = None
                rewards = np.reshape(rewards,(-1,self.num_agents,1))
                dones = np.reshape(dones,(-1,self.num_agents))
                # trunks = np.reshape(trunks,(-1,self.num_agents))
                # dones = np.logical_or(dones,trunks)
                if self.run_mixed_population:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                        values, actions, action_log_probs, \
                        rnn_states_A, rnn_states_B, rnn_states_critic_A, rnn_states_critic_B, policies

                            
                else:
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                            values, actions, action_log_probs, \
                            rnn_states, rnn_states_critic, policies
                
                # insert data into buffer
                insert_start = time.time()
                self.insert(data)
                insert_end = time.time()
                
                TotalInsertTime += insert_end - insert_start
                
            episode_end = time.time()
            print(f"Episode {episode} completed in {episode_end - episode_start:.2f} seconds")
            print(f"Breakdown - Total collect time: {TotalCollectTime:.2f} seconds")
            print(f"Breakdown - Total insert time: {TotalInsertTime:.2f} seconds")
            
            # compute group social influence rewards - modify the buffers rewards
            print("computing social influence at ep:" , episode)
            si_start = time.time()
            graphSimilarityMetrics = self.compute_group_social_influence_rewards(episode) #changed this to be graph equality reward
            graph_similarity_metrics = np.mean(graphSimilarityMetrics)
            si_end = time.time()
            print(f"Social influence computation completed in {si_end - si_start:.2f} seconds")
            print("Average graph similarity metrics: ", graph_similarity_metrics)
            

            # compute return and update network
            print("computing all at ep:" , episode)
            compute_start = time.time()
            self.compute(self.all_args.group_social_influence, self.all_args.group_social_influence_factor)
            compute_end = time.time()
            print(f"Compute completed in {compute_end - compute_start:.2f} seconds")
            if self.no_train_policy:
                print("Not training policy")
                train_infos = self.get_loss()
            else:
                print("train at ep:" , episode)
                train_start = time.time()
                train_infos = self.train()
                train_end = time.time()
                print(f"Training completed in {train_end - train_start:.2f} seconds")
                
            
            if self.evaluate_two_social_influence or self.no_train_si:
                print("not training social influence at ep:" , episode)
                social_train_infos = {}
            else:
                print("train SI at ep:" , episode)
                si_train_start = time.time()
                social_train_infos = self.train_social_influence()
                si_train_end = time.time()
                print(f"Social influence training completed in {si_train_end - si_train_start:.2f} seconds")
                
            social_train_infos.update({"graph_similarity_metrics (used)": graph_similarity_metrics})
            
            
            
            self.finished_training() # Copies last obs to first obs, etc. Used by the train() function originally
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model checkpoint every 5 million steps
            if (total_num_steps % 5000000 < (self.episode_length * self.n_rollout_threads) or episode == episodes - 1):
                print("saving model at ep:" , episode)
                self.save(episode=total_num_steps, episodeLabel=True)
                self.save_si(episode=total_num_steps, episodeLabel=True)
                
            # Updating the central model every million steps
            if (total_num_steps % 1000000 < (self.episode_length * self.n_rollout_threads) or episode == episodes - 1):
                print("saving model at ep:" , episode)
                self.save(episode=episode, episodeLabel=False)
                self.save_si(episode=episode, episodeLabel=False)
                
            

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start_time))))

                if self.env_name == "StarCraft2" or self.env_name == "SMACv2" or self.env_name == "SMAC" or self.env_name == "StarCraft2v2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []                    

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                train_infos.update(social_train_infos)
                
            
                train_infos['total_reward'] = self.buffer.rewards.sum() / (self.n_rollout_threads * self.num_agents * self.episode_length)
                    
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print("Evaluating")
                try:
                    self.eval(total_num_steps)
                except Exception as e:
                    print("Failed to evaluate this round", e)
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
                try:
                    LessGifs = episode % (self.eval_interval * 250) != 0
                    if self.minimal_gifs: 
                        LessGifs = True # getting rid of reciprocity gifs as long as minimal gifs is on
                    self.evaluate_one_episode(total_num_steps, LessGifs=LessGifs) 
                    self.train_one_episode(total_num_steps, LessGifs=LessGifs)
                except Exception as e:
                    print("Failed to evaluate single episode this round", e)
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total run time: {total_time:.2f} seconds")
        ### Fancy colored print statement - green with emoji
        print("\033[92m \U0001F44D \033[0m Finished training")
        
        
    def warmup(self):
        # reset env
        obs,_= self.envs.reset()
        obs = np.reshape(obs,self.buffer.obs[0].shape)
        # replay buffer
        # print(obs.shape)
        if not self.use_centralized_V:
            share_obs = obs.copy()

        # it is using centralized value function 
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = np.ones_like(self.buffer.available_actions[0])
        
    def eval_warmup(self):
        # reset env
        obs, _ = self.eval_envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        # it is using centralized value function 
        self.eval_buffer.share_obs[0] = share_obs.copy()
        self.eval_buffer.obs[0] = obs.copy()
        self.eval_buffer.available_actions[0] = available_actions.copy()
        
    @torch.no_grad()
    def mixed_team_collect(self, step):
        self.trainer_A.prep_rollout()
        
        rnnStates = self.buffer.rnn_states_A
        rnnStatesCritic = self.buffer.rnn_states_critic_A
        value_A, action_A, action_log_prob_A, rnn_state_A, rnn_state_critic_A \
            = self.trainer_A.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(rnnStates[step]),
                                            np.concatenate(rnnStatesCritic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]), deterministic=self.only_use_argmax_action) # Dete
        true_policies_A = self.trainer_A.policy.get_policies(
                                        np.concatenate(self.buffer.obs[step]),
                                        np.concatenate(rnnStates[step]),
                                        np.concatenate(self.buffer.masks[step]),
                                        np.concatenate(self.buffer.available_actions[step]))
        
        # [self.envs, agents, dim]
        values_A = np.array(np.split(_t2n(value_A), self.n_rollout_threads))
        actions_A = np.array(np.split(_t2n(action_A), self.n_rollout_threads))
        action_log_probs_A = np.array(np.split(_t2n(action_log_prob_A), self.n_rollout_threads))
        rnn_states_A = np.array(np.split(_t2n(rnn_state_A), self.n_rollout_threads))
        rnn_states_critic_A = np.array(np.split(_t2n(rnn_state_critic_A), self.n_rollout_threads))
        
        true_policy_A = np.array(np.split(_t2n(true_policies_A), self.n_rollout_threads))
        
        
        #### Trainer B
        self.trainer_B.prep_rollout()
        BufferToUse = self.buffer
        rnnStates = self.buffer.rnn_states_B
        rnnStatesCritic = self.buffer.rnn_states_critic_B

        value_B, action_B, action_log_prob_B, rnn_state_B, rnn_state_critic_B \
            = self.trainer_B.policy.get_actions(np.concatenate(BufferToUse.share_obs[step]),
                                            np.concatenate(BufferToUse.obs[step]),
                                            np.concatenate(rnnStates[step]),
                                            np.concatenate(rnnStatesCritic[step]),
                                            np.concatenate(BufferToUse.masks[step]),
                                            np.concatenate(BufferToUse.available_actions[step]), deterministic=self.only_use_argmax_action) # Dete
        true_policies_B = self.trainer_B.policy.get_policies(
                                        np.concatenate(BufferToUse.obs[step]),
                                        np.concatenate(rnnStates[step]),
                                        np.concatenate(BufferToUse.masks[step]),
                                        np.concatenate(BufferToUse.available_actions[step]))
        
        # [self.envs, agents, dim]
        values_B = np.array(np.split(_t2n(value_B), self.n_rollout_threads))
        actions_B = np.array(np.split(_t2n(action_B), self.n_rollout_threads))
        action_log_probs_B = np.array(np.split(_t2n(action_log_prob_B), self.n_rollout_threads))
        rnn_states_B = np.array(np.split(_t2n(rnn_state_B), self.n_rollout_threads))
        rnn_states_critic_B = np.array(np.split(_t2n(rnn_state_critic_B), self.n_rollout_threads))
        true_policy_B = np.array(np.split(_t2n(true_policies_B), self.n_rollout_threads))
        
        
        values = self.split_for_mixed_teams(values_A, values_B, returnTwoForBuffer=False)
        actions = self.split_for_mixed_teams(actions_A, actions_B, returnTwoForBuffer=False)
        action_log_probs = self.split_for_mixed_teams(action_log_probs_A, action_log_probs_B, returnTwoForBuffer=False)

        # Need to change shape
        true_policy = self.split_for_mixed_teams(true_policy_A, true_policy_B, returnTwoForBuffer=False, TypeDeclaration="policy")

        return values, actions, action_log_probs, rnn_states_A, rnn_states_B, rnn_states_critic_A, rnn_states_critic_B, true_policy
    
    def split_for_mixed_teams(self, values_A, values_B, returnTwoForBuffer=False, TypeDeclaration=None):
        # if TypeDeclaration is None: # action or values
        #     print("Input shapes:", values_A.shape, values_B.shape)
        #     print("Sample input A:", values_A[0])
        #     print("Sample input B:", values_B[0])
        # Mix the two teams by the agents (half from each team)
        OriginalShape = values_A.shape
        # Get the number of agents
        
        # find the first indic that matches self.num_agents
        
        if len(OriginalShape) == 2:
            axisForAgents = 0 # single episode evaluation
        else: 
            axisForAgents = 1 # usual collect split
            
        num_agents = OriginalShape[axisForAgents]
        
        assert num_agents == self.num_agents, f"Number of agents in the buffer (axis 1) {values_A.shape} does not match the number of agents in the environment {self.num_agents}"
        
        # Team B is the wrong size so use helper function to cut it down
        original_values_B = values_B
        if TypeDeclaration is None:
            assert values_A.shape == values_B.shape, f"Shapes of the two teams do not match {values_A.shape} and {values_B.shape}"
        else:
            # There is a need to do fancy stuff to get the correct size
            values_B = values_B
            
            assert values_A.shape == values_B.shape, f"Shapes of the two teams do not match {values_A.shape} and {values_B.shape} even after chopping"

        # Calculate the number of agents to take from each team
        # SplittingPoint = num_agents // 2 # Rounds down so additional agent is from second team

        SplittingPoint = math.floor(num_agents * (self.percent_team_A_agents/100)) # Rounds down so additional agent is from second team

        # print("shapes: ", values_A.shape, values_B.shape)
        # Split values_A and values_B along the agents dimension
        values_A1, values_A2 = np.split(values_A, [SplittingPoint], axis=axisForAgents)
        values_B1, values_B2 = np.split(values_B, [SplittingPoint], axis=axisForAgents)
        
        values_TeamConfigA = np.concatenate((values_A1, values_B2), axis=axisForAgents)
        # values_2 = np.concatenate((values_B1, values_A2), axis=1)
        
        assert values_TeamConfigA.shape == OriginalShape, f"Shape of the mixed values {values_TeamConfigA.shape} does not match the original shape {OriginalShape}"
        
        if returnTwoForBuffer:
            if TypeDeclaration is not None:
                padded_value_A = allThingsTranslationPadding(values_A, TypeDeclaration)
            else: 
                padded_value_A = values_A
            
            assert padded_value_A.shape == original_values_B.shape, f"Shape of the padded values {padded_value_A.shape} does not match the original shape {original_values_B.shape}"
            
            values_A1S, values_A2S = np.split(padded_value_A, [SplittingPoint], axis=axisForAgents)
            values_B1S, values_B2S = np.split(original_values_B, [SplittingPoint], axis=axisForAgents)
            
            values_TeamConfigB = np.concatenate((values_A1S, values_B2S), axis=axisForAgents)
            
            OriginalShapeB = original_values_B.shape
            assert values_TeamConfigB.shape == OriginalShapeB, f"Shape of the mixed values {values_TeamConfigB.shape} does not match the original shape {OriginalShapeB}"
            
            # if TypeDeclaration is None: 
            #     print("Output shapes:", values_TeamConfigA.shape, values_TeamConfigB.shape if returnTwoForBuffer else None)
            #     print("Sample output A:", values_TeamConfigA[0])
            #     print("Sample output B:", values_TeamConfigB[0] if returnTwoForBuffer else None)
    
            
            return values_TeamConfigA, values_TeamConfigB
        
        else:
            
            return values_TeamConfigA

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]), deterministic=self.only_use_argmax_action) # Dete
        # print(action)
        true_policies = self.trainer.policy.get_policies(
                                        np.concatenate(self.buffer.obs[step]),
                                        np.concatenate(self.buffer.rnn_states[step]),
                                        np.concatenate(self.buffer.masks[step]),
                                        np.concatenate(self.buffer.available_actions[step]))
        # print(action)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        
        true_policy = np.array(np.split(_t2n(true_policies), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, true_policy
    
    # def two_buffer_insert(self, data):
    #     assert self.run_mixed_population, "This function should only be called when mixed population is on and the teams are from different environments"
    #     TeamABuf, TeamBBuf = data
    #     for info, BufferToUse in zip([TeamABuf, TeamBBuf], [self.buffer, self.buffer_B]):
    #         obs, share_obs, rewards, dones, infos, available_actions, \
    #         values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = info
            
    #         # Test for whether our available_actions are the right size
    #         # print("Two buffer insert available actions shape:", available_actions.shape)
    #         # print("Two buffer insert available actions sample:", available_actions)
            
    #         dones_env = np.all(dones, axis=1)
            
    #         rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
    #         rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
            
    #         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #         masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

    #         active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
    #         active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
    #         active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

    #         bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
            
    #         if not self.use_centralized_V:
    #             share_obs = obs

    #         BufferToUse.insert(share_obs, obs, rnn_states, rnn_states_critic,
    #             actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions, policies)
    #     return

    def insert(self, data):
        if self.run_mixed_population:
                obs, share_obs, rewards, dones, infos, available_actions, \
                values, actions, action_log_probs, rnn_states_A, rnn_states_B, rnn_states_critic_A, rnn_states_critic_B, policies = data
        else:
            obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = data

        dones_env = np.all(dones, axis=1)
        
        if self.run_mixed_population:
            rnn_states_A[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic_A[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic_A.shape[3:]), dtype=np.float32)
            
            rnn_states_B[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic_B[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic_B.shape[3:]), dtype=np.float32)
            
        else:
            rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        # print(infos)
        bad_masks = None
        
        if not self.use_centralized_V:
            share_obs = obs

        if self.run_mixed_population:
            self.buffer.mixed_insert(share_obs, obs, rnn_states_A, rnn_states_B, rnn_states_critic_A, rnn_states_critic_B,
            actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions, policies)
        else:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions, policies)
        
    def eval_insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.eval_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

        self.eval_buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
            actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions, policies)
        
        

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards (used)"] = np.mean(self.buffer.rewards)
        train_infos["average_step_rewards (game)"] = np.mean(self.buffer.game_rewards)
        train_infos["average_step_rewards (graph_similarity)"] = np.mean(self.buffer.group_social_influence_rewards)
        
        GameRewards = self.buffer.game_rewards 
        WeightedGameRewards = GameRewards * (1-self.all_args.group_social_influence_factor)
        GraphSimilarityRewards = self.buffer.group_social_influence_rewards  # .detach()
        
        if self.all_args.group_social_influence == "CombineSocialGameRwd_withPenalty":
            # MaxGraphSimilarityRewards = np.max(GraphSimilarityRewards) 
            # PenaltyMax = max(0.9, MaxGraphSimilarityRewards)
            PenalizedGraphSimilarityRewards = (1.0-GraphSimilarityRewards) * -1
            # PenalizedGraphSimilarityRewards = (1-GraphSimilarityRewards) * -1
            PenalizedWeightedGraphSimilarityRewards = PenalizedGraphSimilarityRewards * self.all_args.group_social_influence_factor 
            # MyRewards = WeightedGameRewards + PenalizedWeightedGraphSimilarityRewards
            GameRewardsUsed = WeightedGameRewards 
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards
        elif self.all_args.group_social_influence == "CombineSocialGameRwd":
            # MyRewards = WeightedGameRewards + WeightedGraphSimilarityRewards
            WeightedGraphSimilarityRewards = GraphSimilarityRewards * self.all_args.group_social_influence_factor
            GameRewardsUsed = WeightedGameRewards 
            GraphSimilarityRewardsUsed = WeightedGraphSimilarityRewards
        elif self.all_args.group_social_influence == "OnlySocialRwd_withPenalty":
            PenalizedGraphSimilarityRewards = (1-GraphSimilarityRewards) * -1
            MyRewards = PenalizedGraphSimilarityRewards
            GameRewardsUsed = GameRewards * 0 # not used
            GraphSimilarityRewardsUsed = PenalizedGraphSimilarityRewards
        elif self.all_args.group_social_influence == "MultiplyGameSocialRwd":
            # do not use scaling factor here, no penalty
            GameRewardsUsed = GameRewards
            GraphSimilarityRewardsUsed = GraphSimilarityRewards
            MyRewards = GameRewards * GraphSimilarityRewards
        elif self.all_args.group_social_influence == "OnlyConditionalSocial":
            GameRewardsUsed = GameRewards * 0 # not used
            GameRewardMask = GameRewards > 0
            GraphSimilarityRewardsUsed = GraphSimilarityRewards * GameRewardMask 
            # MyRewards = GraphSimilarityRewardsUsed

        elif self.all_args.group_social_influence == "OnlyRandomRewardsUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                0.3, 0.85, size=GraphSimilarityRewards.shape
            )

            PenalizedGraphSimilarityRewards = (1.0 - GraphSimilarityRewards) * -1
            PenalizedWeightedGraphSimilarityRewards = (
                PenalizedGraphSimilarityRewards
                * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = GameRewards * 0  # not used
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )
        elif self.all_args.group_social_influence == "AsymRandomRewardsUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                0.3, 0.85, size=GraphSimilarityRewards.shape
            )

            PenalizedGraphSimilarityRewards = (1.0 - GraphSimilarityRewards) * -1
            PenalizedWeightedGraphSimilarityRewards = (
                PenalizedGraphSimilarityRewards
                * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )
        elif self.all_args.group_social_influence == "SymRandomRewardsUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                -0.5, 0.5, size=GraphSimilarityRewards.shape
            )

            PenalizedWeightedGraphSimilarityRewards = (
                GraphSimilarityRewards * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )

        elif self.all_args.group_social_influence == "BigSymRandomRewardUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                -1, 1, size=GraphSimilarityRewards.shape
            )

            PenalizedWeightedGraphSimilarityRewards = (
                GraphSimilarityRewards * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )
        elif self.all_args.group_social_influence == "MegaSymRandomRewardUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                -5, 5, size=GraphSimilarityRewards.shape
            )

            PenalizedWeightedGraphSimilarityRewards = (
                GraphSimilarityRewards * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )

        elif self.all_args.group_social_influence == "MediumSymRandomRewardUniform":
            # Random Uniform rewards in the shape of the graph similarity rewards from
            # GraphSimilarityRewards = np.random.uniform(-0.4, 0.4, size=GraphSimilarityRewards.shape)
            GraphSimilarityRewards = np.random.uniform(
                -2, 2, size=GraphSimilarityRewards.shape
            )

            PenalizedWeightedGraphSimilarityRewards = (
                GraphSimilarityRewards * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards

            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )

        elif self.all_args.group_social_influence == "RandomRewardGaussian":
            # Random gaussian rewards in the shape of the graph similarity rewards centered around 0 and std of 0.4, take only negative
            GraphSimilarityRewards = np.random.normal(
                0.575, 0.25, size=GraphSimilarityRewards.shape
            )

            PenalizedGraphSimilarityRewards = (1.0 - GraphSimilarityRewards) * -1
            PenalizedWeightedGraphSimilarityRewards = (
                PenalizedGraphSimilarityRewards
                * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards
            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )
        elif self.all_args.group_social_influence == "RandomRewardMimic":
            # based off the original graph similarity rewards's mean and std
            GraphSimilarityRewards = np.random.normal(
                np.mean(GraphSimilarityRewards),
                np.std(GraphSimilarityRewards),
                size=GraphSimilarityRewards.shape,
            )
            PenalizedGraphSimilarityRewards = (1.0 - GraphSimilarityRewards) * -1
            PenalizedWeightedGraphSimilarityRewards = (
                PenalizedGraphSimilarityRewards
                * self.all_args.group_social_influence_factor
            )
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = PenalizedWeightedGraphSimilarityRewards
            train_infos["average_step_rewards (weighted random)"] = np.mean(
                GraphSimilarityRewardsUsed
            )
        elif self.all_args.group_social_influence == "OnlyWeightedGameRwd":
            GameRewardsUsed = WeightedGameRewards
            GraphSimilarityRewardsUsed = GraphSimilarityRewards * 0  # not used
        elif self.all_args.group_social_influence == "OnlyGameRwd":
            GameRewardsUsed = GameRewards
            GraphSimilarityRewardsUsed = GraphSimilarityRewards * 0 # not used
        elif self.all_args.group_social_influence == "OnlySocialRwd":
            print(
                "WARNING: OnlySocialRwd is not a valid reward for training. You should be doing OnlySocialRwd_withPenalty."
            )
            GameRewardsUsed = GameRewards * 0  # not used
            GraphSimilarityRewardsUsed = GraphSimilarityRewards
        elif "ConstantPenalty" in self.all_args.group_social_influence:
            CurrentRange = 1
            # Penalty
            Penalty = 0
            if "ConstantPenalty05" in self.all_args.group_social_influence:
                Penalty = -0.5
            elif "ConstantPenalty1" in self.all_args.group_social_influence:
                Penalty = -1
            GameRewardsUsed = GameRewards + Penalty

            # Noise
            if "Noise" in self.all_args.group_social_influence:
                CurrentRange = 2
                Noise = np.random.uniform(-0.5, 0.5, size=GraphSimilarityRewards.shape)
                GameRewardsUsed = GameRewards + Noise

            if "Similarity" in self.all_args.group_social_influence:
                SocialReward = GraphSimilarityRewards
                CurrentRange = 2
                if "Similarity05" in self.all_args.group_social_influence:
                    # SocialReward = (0.5 - GraphSimilarityRewards) * -1
                    SocialReward = GraphSimilarityRewards - 0.5
                    CurrentRange = 2

                if "Proportion" in self.all_args.group_social_influence:
                    SocialReward = (
                        SocialReward * self.all_args.group_social_influence_factor
                    )
                    GameRewardsUsed = GameRewardsUsed * (
                        1 - self.all_args.group_social_influence_factor
                    )

                # MyRewards = GameRewardsUsed + SocialReward
                GraphSimilarityRewardsUsed = SocialReward
            else:
                GraphSimilarityRewardsUsed = GraphSimilarityRewards * 0

            if "Rescale" in self.all_args.group_social_influence:
                if "Rescale05" in self.all_args.group_social_influence:
                    desiredRange = 0.5
                elif "Rescale1" in self.all_args.group_social_influence:
                    desiredRange = 1
                elif "Rescale2" in self.all_args.group_social_influence:
                    desiredRange = 2

                MyRange = desiredRange + desiredRange  # should be 1 and 2
                if MyRange != CurrentRange:
                    GraphSimilarityRewardsUsed = GraphSimilarityRewardsUsed * (
                        MyRange / CurrentRange
                    )
                    GameRewardsUsed = GameRewardsUsed * (MyRange / CurrentRange)

        else:
            GameRewardsUsed = GameRewards
            GraphSimilarityRewardsUsed = GraphSimilarityRewards

        train_infos["average_step_rewards (weighted game)"] = np.mean(GameRewardsUsed)
        train_infos["average_step_rewards (weighted graph_similarity)"] = np.mean(
            GraphSimilarityRewardsUsed
        )

        for k, v in train_infos.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)
                
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        eval_timestep_rewards = []
        # one_episode_rewards = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)] # one_episode_rewards should be a list containing lists of episode rewards per thread. 

        eval_obs, _= self.eval_envs.reset()
        eval_obs_B = eval_obs
        eval_available_actions_B = None

        eval_share_obs = eval_obs.copy()
        if self.run_mixed_population:
            eval_rnn_states_A = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_rnn_states_B = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        else:
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks_A = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_masks_B = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        counter = 0
        while True:
            counter+=1
            if self.run_mixed_population:
                self.trainer_A.prep_rollout()
                self.trainer_B.prep_rollout()
            else:
                self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                # Not used for MAPPO 
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(None),
                                            deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            else:
                if self.run_mixed_population:
                    eval_actions_A, eval_rnn_states_A = \
                    self.trainer_A.policy.act(np.stack(eval_obs),
                                            np.concatenate(eval_rnn_states_A),
                                            np.concatenate(eval_masks_A),
                                            None,
                                            deterministic=False)
                    eval_actions_B, eval_rnn_states_B = \
                    self.trainer_B.policy.act(np.stack(eval_obs_B),
                                            np.concatenate(eval_rnn_states_B),
                                            np.concatenate(eval_masks_B),
                                            None,
                                            deterministic=False)
                    
                    eval_actions_A = np.array(np.split(_t2n(eval_actions_A), self.n_eval_rollout_threads))
                    eval_actions_B = np.array(np.split(_t2n(eval_actions_B), self.n_eval_rollout_threads))
                    eval_actions = self.split_for_mixed_teams(eval_actions_A, eval_actions_B) # The one that is actually used
                    # print("Eval funct, actions shape:", eval_actions.shape)
                    # print("Eval funct, actions team a shape:", eval_actions_A.shape)
                    # print("Eval funct, actions team b shape:", eval_actions_B.shape)
                    # print("Eval funct, actions sample:", eval_actions)
                    # print("Eval funct, actions team a sample:", eval_actions_A)
                    # print("Eval funct, actions team b sample:", eval_actions_B)
                    
                    eval_rnn_states_A = np.array(np.split(_t2n(eval_rnn_states_A), self.n_eval_rollout_threads))
                    eval_rnn_states_B = np.array(np.split(_t2n(eval_rnn_states_B), self.n_eval_rollout_threads))
                else:
                    eval_actions, eval_rnn_states = \
                        self.trainer.policy.act(np.stack(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                None,
                                                deterministic=False)
                        
                    eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            # observations, rewards, terminations, truncations, infos
            # print(eval_actions.shape)
            eval_actions= np.reshape(eval_actions,(-1))
            eval_obs, eval_rewards, eval_dones,eval_truncations, eval_infos = self.eval_envs.step(eval_actions)
            # print(eval_dones.shape)
            # print(f"Step rewards: {eval_rewards}")  # Debug print
            eval_truncations = np.reshape(eval_truncations,(-1,self.num_agents))
            eval_dones = np.reshape(eval_dones,(-1,self.num_agents))
            
            for i in range(self.n_eval_rollout_threads):
                one_episode_rewards[i].append(eval_rewards[i])

            eval_dones_env = np.logical_or(np.all(eval_dones, axis=1),np.all(eval_truncations,axis=1))
            # print(eval_dones_env.shape)
            # print(eval_masks.shape)
            # print([eval_dones_env == True])
            if self.run_mixed_population:
                eval_rnn_states_A[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_rnn_states_B[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            else:
                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)


            # if counter>498:
                # print(eval_dones_env)
            # eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            # print(counter)
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    # print("EVAL",eval_episode)
                    # print(f"Episode {eval_episode} rewards: {one_episode_rewards[eval_i]}")  # Debug print
                    eval_episode_rewards.append(np.sum(one_episode_rewards[eval_i]))                    # one_episode_rewards = []
                    timestep_reward = np.mean(one_episode_rewards[eval_i])
                    eval_timestep_rewards.append(timestep_reward)
                    one_episode_rewards[eval_i] = []
                    # if eval_infos[eval_i][0]['won']:
                    #     eval_battles_won += 1


            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_timestep_rewards = np.array(eval_timestep_rewards)
                # print(f"Eval episode rewards: {eval_episode_rewards}")  # Debug print
                eval_env_infos = {'eval_average_episode_rewards (game reward)': eval_episode_rewards, 'eval_average_step_rewards (game reward)': eval_timestep_rewards}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))

                metrics = {"eval_win_rate": eval_win_rate}
                
                # Log metrics
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break

    def evaluate_one_episode(self, total_num_steps, LessGifs=False):
        """Doing evaluation with one environment and one episode for rendering purpose"""
        episode_reward = 0
        episode_steps = 0
        done = False
        battle_won = False
        
        MyObs = []
        MyActions = []
        MyRewards = []
        MyPolicies = []

        eval_obs, _ = self.single_eval_envs.reset()
        
        eval_share_obs = np.array(eval_obs)
        eval_available_actions = None
        eval_obs = np.array(eval_obs)
        
        # unsqueeze at 0
        eval_obs = np.expand_dims(eval_obs, axis=0)
        eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
        eval_available_actions = None
        
        eval_obs_B = eval_obs
        eval_share_obs_B = eval_share_obs
        eval_available_actions_B = None
        
        frames = []  # List to save the frames
        
        if self.run_mixed_population:
            eval_rnn_states_A = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_rnn_states_B = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        else:
            eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        positionOfAgents = [] # List to save the position of agents
        
        MyObs.append(eval_obs)

        while not done:
            with torch.no_grad():
                if self.run_mixed_population:
                    self.trainer_A.prep_rollout()
                    self.trainer_B.prep_rollout()
                    true_policies_A = self.trainer_A.policy.get_policies(
                                np.concatenate(eval_obs),
                                np.concatenate(eval_rnn_states_A),
                                np.concatenate(eval_masks),
                                np.concatenate(eval_available_actions)) 
                    
                    true_policies_B = self.trainer_B.policy.get_policies(
                                np.concatenate(eval_obs_B),
                                np.concatenate(eval_rnn_states_B),
                                np.concatenate(eval_masks),
                                np.concatenate(eval_available_actions_B)) 
                    true_policies = self.split_for_mixed_teams(true_policies_A, true_policies_B) # We return in Team A format
                    
                else:
                    self.trainer.prep_rollout()
                    true_policies = self.trainer.policy.get_policies(
                                np.concatenate(eval_obs),
                                np.concatenate(eval_rnn_states),
                                np.concatenate(eval_masks),
                                None)                  
                if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                    eval_actions, eval_rnn_states = \
                        self.trainer.policy.act(np.concatenate(eval_share_obs),
                                                np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                None,
                                                deterministic=False)

                else:
                    if self.run_mixed_population:
                        eval_actions_A, eval_rnn_states_A = \
                            self.trainer_A.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states_A),
                                                    np.concatenate(eval_masks),
                                                    None,
                                                    deterministic=True)
                        eval_actions_B, eval_rnn_states_B = \
                            self.trainer_B.policy.act(np.concatenate(eval_obs_B),
                                                    np.concatenate(eval_rnn_states_B),
                                                    np.concatenate(eval_masks),
                                                    None,
                                                    deterministic=True)
                            
                        eval_actions = self.split_for_mixed_teams(eval_actions_A, eval_actions_B) # actions should be integer so no size difference
                        
                    else:
                        eval_actions, eval_rnn_states = \
                            self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    None,
                                                    deterministic=True)

                # Don't split because this is single environment
                
                # Save the current frame
                try:
                    frame = self.single_eval_envs.render()
                    # print(frame)
                    # print("hello")
                    frames.append(frame)
                except Exception as e:
                    if episode_steps == 0:
                        print(e)
                        print("Cannot render here")
                        print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                        traceback.print_exc()
                    try:
                        if self.single_eval_envs.render is not None:
                            frame = self.single_eval_envs.render()
                            frames.append(frame)
                    except Exception as e:
                        print(e)
                        print("Second attempt to render failed")
                        print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                        traceback.print_exc()
                
                # local_obs, global_state, rewards, dones, infos, avail_actions
                # currentPosition = np.zeros((self.num_agents, 2), dtype=np.float32)
                # for agent_id in range(self.num_agents):
                #     xpos = self.single_eval_envs.get_unit_by_id(agent_id).pos.x / self.single_eval_envs.map_x
                #     ypos = self.single_eval_envs.get_unit_by_id(agent_id).pos.y / self.single_eval_envs.map_y
                #     currentPosition[agent_id] = [xpos, ypos]
                # positionOfAgents.append(currentPosition)
                # Obser reward and next obs
                # eval_dones is per agent
                # print(eval_actions.shape)
                eval_actions = eval_actions.cpu().numpy()
                eval_actions = np.reshape(eval_actions,(-1))
                eval_obs, eval_rewards, eval_dones,eval_truncations, eval_infos = self.single_eval_envs.step(eval_actions)
                # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                if not self.run_mixed_population:
                    true_policies = true_policies.cpu().numpy()
                # print(eval_dones.shape)
                eval_share_obs = eval_obs.copy()
                eval_available_actions = None
                
                eval_truncations = np.reshape(eval_truncations,(-1,self.num_agents))
                eval_dones = np.reshape(eval_dones,(-1,self.num_agents))
            
            

                eval_dones = np.logical_or(np.all(eval_dones, axis=1),np.all(eval_truncations,axis=1))
                eval_obs = np.expand_dims(eval_obs, axis=0)
                eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
                eval_rewards = np.expand_dims(eval_rewards, axis=0)
                # eval_available_actions = np.expand_dims(eval_available_actions, axis=0)
                eval_actions = np.expand_dims(eval_actions, axis=0)
                
                true_policies = np.expand_dims(true_policies, axis=0)
                
                MyRewards.append(eval_rewards)
                MyObs.append(eval_obs) # why is this after action and not before?
                MyActions.append(eval_actions)
                MyPolicies.append(true_policies)

                episode_steps += 1
                # print(eval_dones.shape)
                done = np.all(eval_dones)
                # print(done.shape)                                 
                # print(eval_rnn_states.shape)
                # print(env_done.shape)
                ###
                env_done = np.expand_dims(done, axis=0)
                # print(env_done.shape)
                if self.run_mixed_population:
                    eval_rnn_states_A = np.array(np.split(_t2n(eval_rnn_states_A), self.n_eval_rollout_threads))
                    eval_rnn_states_B = np.array(np.split(_t2n(eval_rnn_states_B), self.n_eval_rollout_threads))
                    eval_rnn_states_A[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    eval_rnn_states_B[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                else:
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                    # print(eval_rnn_states.shape)
                    # print(env_done.shape)
                    # print(eval_rnn_states.shape)
                    if env_done.sum() == 1:
                        eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)


                eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                if env_done:
                    eval_masks[env_done == True] = np.zeros((1, self.num_agents, 1), dtype=np.float32)
                # eval_masks[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, 1), dtype=np.float32)
                
                # print("Dones", done)

        data_dic = {}
        if done and 'battle_won' in eval_infos:
            battle_won = eval_infos['battle_won']
            
        # Prepare MyObs, MyActions, MyRewards into np array instead of list
        MyObs = np.stack(MyObs, axis=0)
        MyActions = np.stack(MyActions, axis=0)
        MyActions = np.expand_dims(MyActions, axis=-1)
        MyRewards = np.stack(MyRewards, axis=0)
        MyPolicies = np.stack(MyPolicies, axis=0)

        #stack positionOfAgents with dim 0 being new
        # positionOfAgents = np.stack(positionOfAgents, axis=0)

        # Save the last frame
        try:
            frame = self.single_eval_envs.render()
            # print(frame)
            frames.append(frame)
        except Exception as e:
            if episode_steps == 0:
                print(e)
                print("Cannot render here")
                print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                traceback.print_exc()
            try:
                if self.single_eval_envs.render is not None:
                    # self.single_eval_envs.renderer = StarCraft2Renderer(self.single_eval_envs, "rgb_array")
                    frame = self.single_eval_envs.render()
                    frames.append(frame)
            except Exception as e:
                print(e)
                print("Second attempt to render failed")
                print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                traceback.print_exc()
            
        # Save the frames as a gif
        render_save_path = None
        if len(frames) > 0:
            # frames = self.social_influence_network.annotate_frames(frames, MyActions, MyPolicies) # This is seperate from the social influence network so it does not matter which one is used
            # Turn off policy annotation since we are doing individual evaluation
            # frames = self.social_influence_network.annotate_frames(frames, MyActions, None) # This is seperate from the social influence network so it does not matter which one is used
            render_save_path = f'{self.gifs_dir}/Eval-Render-SMAC_Ep{total_num_steps}.gif'
            imageio.mimwrite(uri=render_save_path, ims=frames, fps=1)
            data_dic["single_ep_eval/SMAC_video"] = wandb.Video(render_save_path)
        
        

        # print("Position of Agents: ", positionOfAgents.shape)
        
        ## Do social influence evaluation here
        
        
        metrics = {}
        
        if self.evaluate_two_social_influence:
            try:
                metrics = self.compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(MyObs, MyActions, MyRewards, total_num_steps=total_num_steps, positionOfAgents=positionOfAgents, minimal_gifs=LessGifs, frames=frames, policies = MyPolicies)
                
            except Exception as e:
                print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                traceback.print_exc()

        else:
            try:
                # Calls base runner that handles A vs B conversion
                metrics = self.compute_group_social_influence_rewards_evaluation_single_episode(MyObs, MyActions, MyRewards, total_num_steps=total_num_steps, positionOfAgents=positionOfAgents, minimal_gifs=LessGifs, frames=frames, policies = MyPolicies)

            except Exception as e:
                print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                traceback.print_exc()
        
        
        # social influence rewards
        
        episode_reward = np.mean(MyRewards)
        
        data_dic.update({
                    "single_ep_eval/battle_won": battle_won,
                    "single_ep_eval/episode_reward": episode_reward,
                    "single_ep_eval/episode_steps": episode_steps,
                    })
        data_dic.update(metrics)
        
        

        wandb.log(data_dic, step=total_num_steps)
                
        # SI_reward = metrics.get("single_ep_eval/social_influence_metric_mean", "N/A")
        # SI_reward = metrics.get("Group Social Influence Rewards (Cosine Similarity)", "N/A")
        SI_reward = metrics.get("single_ep_eval/graph_similarity-Cosine-NoPruning (Ep Mean)", "N/A")
        if SI_reward == "N/A":
            print(metrics.keys())   

        print(f"Single Ep Evaluation - Episode reward: {episode_reward}, Episode steps: {episode_steps} - Social Influence Reward: {SI_reward}")
        return episode_reward, battle_won
    
    def train_one_episode(self, total_num_steps, LessGifs=False):
            """Doing evaluation with one environment and one episode for rendering purpose"""
            episode_reward = 0
            episode_steps = 0
            done = False
            battle_won = False
            
            MyObs = []
            MyActions = []
            MyRewards = []
            MyPolicies = []

            eval_obs, _ = self.single_eval_envs.reset()
            
            eval_share_obs = np.array(eval_obs)
            eval_available_actions = None
            eval_obs = np.array(eval_obs)
            
            # unsqueeze at 0
            eval_obs = np.expand_dims(eval_obs, axis=0)
            eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
            eval_available_actions = None
            
            eval_obs_B = eval_obs
            eval_share_obs_B = eval_share_obs
            eval_available_actions_B = None
            
            frames = []  # List to save the frames
            
            if self.run_mixed_population:
                eval_rnn_states_A = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_rnn_states_B = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            else:
                eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            positionOfAgents = [] # List to save the position of agents
            
            MyObs.append(eval_obs)

            while not done:
                with torch.no_grad():
                    if self.run_mixed_population:
                        self.trainer_A.prep_rollout()
                        self.trainer_B.prep_rollout()
                        true_policies_A = self.trainer_A.policy.get_policies(
                                    np.concatenate(eval_obs),
                                    np.concatenate(eval_rnn_states_A),
                                    np.concatenate(eval_masks),
                                    None) 
                        
                        true_policies_B = self.trainer_B.policy.get_policies(
                                    np.concatenate(eval_obs_B),
                                    np.concatenate(eval_rnn_states_B),
                                    np.concatenate(eval_masks),
                                    None) 
                        true_policies = self.split_for_mixed_teams(true_policies_A.cpu().numpy(), true_policies_B.cpu().numpy()) # We return in Team A format
                        
                    else:
                        self.trainer.prep_rollout()
                        true_policies = self.trainer.policy.get_policies(
                                    np.concatenate(eval_obs),
                                    np.concatenate(eval_rnn_states),
                                    np.concatenate(eval_masks),
                                    None)                  
                    if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                        eval_actions, eval_rnn_states = \
                            self.trainer.policy.act(np.concatenate(eval_share_obs),
                                                    np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    np.concatenate(eval_available_actions),
                                                    deterministic=False)

                    else:
                        if self.run_mixed_population:
                            eval_actions_A, eval_rnn_states_A = \
                                self.trainer_A.policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states_A),
                                                        np.concatenate(eval_masks),
                                                        None,
                                                        deterministic=False)
                            eval_actions_B, eval_rnn_states_B = \
                                self.trainer_B.policy.act(np.concatenate(eval_obs_B),
                                                        np.concatenate(eval_rnn_states_B),
                                                        np.concatenate(eval_masks),
                                                        None,
                                                        deterministic=False)
                                
                            eval_actions = self.split_for_mixed_teams(eval_actions_A.cpu(), eval_actions_B.cpu()) # actions should be integer so no size difference
                            
                        else:
                            eval_actions, eval_rnn_states = \
                                self.trainer.policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks),
                                                        None,
                                                        deterministic=False)
                            eval_actions = eval_actions.cpu().numpy()

                    # Don't split because this is single environment
                    
                    # Save the current frame
                    try:
                        frame = self.single_eval_envs.render()
                        # print(frame)
                        # print("hello")
                        frames.append(frame)
                    except Exception as e:
                        if episode_steps == 0:
                            print(e)
                            print("Cannot render here")
                            print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                            traceback.print_exc()
                        try:
                            if self.single_eval_envs.render is not None:
                                frame = self.single_eval_envs.render()
                                frames.append(frame)
                        except Exception as e:
                            print(e)
                            print("Second attempt to render failed")
                            print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                            traceback.print_exc()
                    
                    # local_obs, global_state, rewards, dones, infos, avail_actions
                    # currentPosition = np.zeros((self.num_agents, 2), dtype=np.float32)
                    # for agent_id in range(self.num_agents):
                    #     xpos = self.single_eval_envs.get_unit_by_id(agent_id).pos.x / self.single_eval_envs.map_x
                    #     ypos = self.single_eval_envs.get_unit_by_id(agent_id).pos.y / self.single_eval_envs.map_y
                    #     currentPosition[agent_id] = [xpos, ypos]
                    # positionOfAgents.append(currentPosition)
                    # Obser reward and next obs
                    # eval_dones is per agent
                    # print(eval_actions.shape)
                    # eval_actions = eval_actions.cpu().numpy()
                    eval_actions = np.reshape(eval_actions,(-1))
                    eval_obs, eval_rewards, eval_dones,eval_truncations, eval_infos = self.single_eval_envs.step(eval_actions)
                    # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                    if not self.run_mixed_population:
                        true_policies = true_policies.cpu().numpy()
                    # print(eval_dones.shape)
                    eval_share_obs = eval_obs.copy()
                    eval_available_actions = None
                    
                    eval_truncations = np.reshape(eval_truncations,(-1,self.num_agents))
                    eval_dones = np.reshape(eval_dones,(-1,self.num_agents))
                
                

                    eval_dones = np.logical_or(np.all(eval_dones, axis=1),np.all(eval_truncations,axis=1))
                    eval_obs = np.expand_dims(eval_obs, axis=0)
                    eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
                    eval_rewards = np.expand_dims(eval_rewards, axis=0)
                    # eval_available_actions = np.expand_dims(eval_available_actions, axis=0)
                    eval_actions = np.expand_dims(eval_actions, axis=0)
                    
                    true_policies = np.expand_dims(true_policies, axis=0)
                    
                    MyRewards.append(eval_rewards)
                    MyObs.append(eval_obs) # why is this after action and not before?
                    MyActions.append(eval_actions)
                    MyPolicies.append(true_policies)

                    episode_steps += 1
                    # print(eval_dones.shape)
                    done = np.all(eval_dones)
                    # print(done.shape)                                 
                    # print(eval_rnn_states.shape)
                    # print(env_done.shape)
                    ###
                    env_done = np.expand_dims(done, axis=0)
                    # print(env_done.shape)
                    if self.run_mixed_population:
                        eval_rnn_states_A = np.array(np.split(_t2n(eval_rnn_states_A), self.n_eval_rollout_threads))
                        eval_rnn_states_B = np.array(np.split(_t2n(eval_rnn_states_B), self.n_eval_rollout_threads))
                        eval_rnn_states_A[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                        eval_rnn_states_B[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    else:
                        eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                        # print(eval_rnn_states.shape)
                        # print(env_done.shape)
                        # print(eval_rnn_states.shape)
                        if env_done.sum() == 1:
                            eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)


                    eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                    if env_done:
                        eval_masks[env_done == True] = np.zeros((1, self.num_agents, 1), dtype=np.float32)
                    # eval_masks[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, 1), dtype=np.float32)
                    
                    # print("Dones", done)

            data_dic = {}
            if done and 'battle_won' in eval_infos:
                battle_won = eval_infos['battle_won']
                
            # Prepare MyObs, MyActions, MyRewards into np array instead of list
            MyObs = np.stack(MyObs, axis=0)
            MyActions = np.stack(MyActions, axis=0)
            MyActions = np.expand_dims(MyActions, axis=-1)
            MyRewards = np.stack(MyRewards, axis=0)
            MyPolicies = np.stack(MyPolicies, axis=0)

            #stack positionOfAgents with dim 0 being new
            # positionOfAgents = np.stack(positionOfAgents, axis=0)

            # Save the last frame
            try:
                frame = self.single_eval_envs.render()
                # print(frame)
                frames.append(frame)
            except Exception as e:
                if episode_steps == 0:
                    print(e)
                    print("Cannot render here")
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
                try:
                    if self.single_eval_envs.render is not None:
                        # self.single_eval_envs.renderer = StarCraft2Renderer(self.single_eval_envs, "rgb_array")
                        frame = self.single_eval_envs.render()
                        frames.append(frame)
                except Exception as e:
                    print(e)
                    print("Second attempt to render failed")
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
                
            # Save the frames as a gif
            render_save_path = None
            if len(frames) > 0:
                # frames = self.social_influence_network.annotate_frames(frames, MyActions, MyPolicies) # This is seperate from the social influence network so it does not matter which one is used
                # Turn off policy annotation since we are doing individual evaluation
                # frames = self.social_influence_network.annotate_frames(frames, MyActions, None) # This is seperate from the social influence network so it does not matter which one is used
                render_save_path = f'{self.gifs_dir}/Train-Render-SMAC_Ep{total_num_steps}.gif'
                imageio.mimwrite(uri=render_save_path, ims=frames, fps=1)
                data_dic["single_ep_train/SMAC_video"] = wandb.Video(render_save_path)
            
            

            # print("Position of Agents: ", positionOfAgents.shape)
            
            ## Do social influence evaluation here
            
            
            metrics = {}
            
            if self.evaluate_two_social_influence:
                try:
                    metrics = self.compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(MyObs, MyActions, MyRewards, total_num_steps=total_num_steps, positionOfAgents=positionOfAgents, minimal_gifs=LessGifs, frames=frames, policies = MyPolicies)
                    
                except Exception as e:
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()

            else:
                try:
                    # Calls base runner that handles A vs B conversion
                    metrics = self.compute_group_social_influence_rewards_evaluation_single_episode(MyObs, MyActions, MyRewards, total_num_steps=total_num_steps, positionOfAgents=positionOfAgents, minimal_gifs=LessGifs, frames=frames, policies = MyPolicies)

                except Exception as e:
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
            
            
            # social influence rewards
            
            episode_reward = np.mean(MyRewards)
            
            data_dic.update({
                        "single_ep_train/battle_won": battle_won,
                        "single_ep_train/episode_reward": episode_reward,
                        "single_ep_train/episode_steps": episode_steps,
                        })
            data_dic.update(metrics)
            
            

            wandb.log(data_dic, step=total_num_steps)
                    
            # SI_reward = metrics.get("single_ep_eval/social_influence_metric_mean", "N/A")
            # SI_reward = metrics.get("Group Social Influence Rewards (Cosine Similarity)", "N/A")
            SI_reward = metrics.get("single_ep_train/graph_similarity-Cosine-NoPruning (Ep Mean)", "N/A")
            if SI_reward == "N/A":
                print(metrics.keys())   

            print(f"Single Ep Evaluation Train - Episode reward: {episode_reward}, Episode steps: {episode_steps} - Social Influence Reward: {SI_reward}")
            return episode_reward, battle_won