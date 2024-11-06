import time
import traceback
from functools import reduce

import imageio
import numpy as np
import torch
import wandb
from onpolicy.runner.shared.base_runner import Runner

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
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            print("##### Episode: ", episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = self.collect(step)
                    
                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, policies
                
                # insert data into buffer
                self.insert(data)
                
                
            # compute group social influence rewards - modify the buffers rewards
            print("computing social influence at ep:" , episode)
            self.compute_group_social_influence_rewards(episode)
                
            # Modify rewards here if social influence is used
            # if self.all_args.group_social_influence:
            #     rewards = self.envs.social_influence_rewards(rewards)

            # compute return and update network
            print("computing all at ep:" , episode)
            self.compute(self.all_args.group_social_influence, self.all_args.group_social_influence_factor)
            if self.no_train_policy:
                print("Not training policy")
                train_infos = self.get_loss()
            else:
                print("train at ep:" , episode)
                train_infos = self.train()
                
            
            if self.evaluate_two_social_influence:
                print("not training social influence at ep:" , episode)
                social_train_infos = {}
            else:
                print("train SI at ep:" , episode)
                social_train_infos = self.train_social_influence()
            
            self.finished_training() # Copies last obs to first obs, etc. Used by the train() function originally
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           
            # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     print("saving model at ep:" , episode)
            #     self.save()
            #     self.save_si()

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
                                int(total_num_steps / (end - start))))

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
                        self.summary_writer.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                    
                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer.active_masks.shape)) 
                
                train_infos.update(social_train_infos)
                
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
                    LessGifs = False
                    if self.minimal_gifs: 
                        LessGifs = episode % (self.eval_interval * 250) != 0
                    self.evaluate_one_episode(total_num_steps, LessGifs=LessGifs)
                except Exception as e:
                    print("Failed to evaluate single episode this round", e)
                    print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                    traceback.print_exc()
            # if episode % self.eval_interval == 5 and self.use_eval:
            #     self.clear_gifs() # Needs to stagger or wandb will have bug
                                
    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        # it is using centralized value function 
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()
        
    def eval_warmup(self):
        # reset env
        obs, share_obs, available_actions = self.eval_envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        # it is using centralized value function 
        self.eval_buffer.share_obs[0] = share_obs.copy()
        self.eval_buffer.obs[0] = obs.copy()
        self.eval_buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.rnn_states[step]),
                                            np.concatenate(self.buffer.rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]),
                                            np.concatenate(self.buffer.available_actions[step]))
        true_policies = self.trainer.policy.get_policies(np.concatenate(self.buffer.share_obs[step]),
                                        np.concatenate(self.buffer.obs[step]),
                                        np.concatenate(self.buffer.rnn_states[step]),
                                        np.concatenate(self.buffer.rnn_states_critic[step]),
                                        np.concatenate(self.buffer.masks[step]),
                                        np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))
        
        true_policy = np.array(np.split(_t2n(true_policies), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, true_policy
    
    @torch.no_grad()
    def eval_collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.eval_buffer.share_obs[step]),
                                            np.concatenate(self.eval_buffer.obs[step]),
                                            np.concatenate(self.eval_buffer.rnn_states[step]),
                                            np.concatenate(self.eval_buffer.rnn_states_critic[step]),
                                            np.concatenate(self.eval_buffer.masks[step]),
                                            np.concatenate(self.eval_buffer.available_actions[step]))
        true_policies = self.trainer.policy.get_policies(np.concatenate(self.eval_buffer.share_obs[step]),
                                        np.concatenate(self.eval_buffer.obs[step]),
                                        np.concatenate(self.eval_buffer.rnn_states[step]),
                                        np.concatenate(self.eval_buffer.rnn_states_critic[step]),
                                        np.concatenate(self.eval_buffer.masks[step]),
                                        np.concatenate(self.eval_buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_eval_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_eval_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_eval_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_eval_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_eval_rollout_threads))
        
        true_policy = np.array(np.split(_t2n(true_policies), self.n_eval_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, true_policy

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, policies = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs

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
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.summary_writer.add_scalars(k, {k: v}, total_num_steps)
                
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        # print("PROP eval_obs size: ", eval_obs.shape)
        # print("PROP eval_share_obs size: ", eval_share_obs.shape)
        # print("PROP eval_available_actions size: ", eval_available_actions.shape)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_share_obs),
                                            np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            else:
                eval_actions, eval_rnn_states = \
                    self.trainer.policy.act(np.concatenate(eval_obs),
                                            np.concatenate(eval_rnn_states),
                                            np.concatenate(eval_masks),
                                            np.concatenate(eval_available_actions),
                                            deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)
            
            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1
                        

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    # social influence rewards
                    # meanSIRwd = np.mean(perAgentSIRwd)
                     
                    wandb.log({"eval_win_rate": eval_win_rate,
                            #    "eval_social_influence_reward": meanSIRwd,
                            #    "eval_social_influence_graph": wandb.Image(gif_save_path)
                               }, step=total_num_steps)
                else:
                    self.summary_writer.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
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

        eval_obs, eval_share_obs, eval_available_actions = self.single_eval_envs.reset()
        # print sizes of eval_obs, eval_share_obs, eval_available_actions
        # print("eval_obs size: ", eval_obs.shape)
        # print("eval_share_obs size: ", eval_share_obs.shape)
        # print("eval_available_actions size: ", eval_available_actions.shape)
        
        # np.array
        eval_obs = np.array(eval_obs)
        eval_share_obs = np.array(eval_share_obs)
        eval_available_actions = np.array(eval_available_actions)
        
        # unsqueeze at 0
        eval_obs = np.expand_dims(eval_obs, axis=0)
        eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
        eval_available_actions = np.expand_dims(eval_available_actions, axis=0)
        
        # print("eval_obs size: ", eval_obs.shape)
        # print("eval_share_obs size: ", eval_share_obs.shape)
        # print("eval_available_actions size: ", eval_available_actions.shape)
        
        frames = []  # List to save the frames
        
        eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        positionOfAgents = [] # List to save the position of agents
        
        MyObs.append(eval_obs)

        while not done:
            with torch.no_grad():
                self.trainer.prep_rollout()
                # put this here because if you do 'act' rnn_states will be updated
                true_policies = self.trainer.policy.get_policies(
                            np.concatenate(eval_obs),
                            np.concatenate(eval_rnn_states),
                            np.concatenate(eval_masks),
                            np.concatenate(eval_available_actions)) 
                if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
                    eval_actions, eval_rnn_states = \
                        self.trainer.policy.act(np.concatenate(eval_share_obs),
                                                np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)

                else:
                    eval_actions, eval_rnn_states = \
                        self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                np.concatenate(eval_available_actions),
                                                deterministic=True)

                        
                # Don't split because this is single environment
                
                # Save the current frame
                try:
                    frame = self.single_eval_envs.render(mode="rgb_array")
                    frames.append(frame)
                except Exception as e:
                    if episode_steps == 0:
                        print(e)
                        print("Cannot render here")
                        print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                        traceback.print_exc()
                    try:
                        if self.single_eval_envs.renderer is not None:
                            # self.single_eval_envs.renderer = StarCraft2Renderer(self.single_eval_envs, "rgb_array")
                            frame = self.single_eval_envs.renderer.render("rgb_array")
                            frames.append(frame)
                    except Exception as e:
                        print(e)
                        print("Second attempt to render failed")
                        print(f"Exception type: {type(e).__name__}, Exception message: {str(e)}")
                        traceback.print_exc()
                # local_obs, global_state, rewards, dones, infos, avail_actions
                currentPosition = np.zeros((self.num_agents, 2), dtype=np.float32)
                for agent_id in range(self.num_agents):
                    xpos = self.single_eval_envs.get_unit_by_id(agent_id).pos.x / self.single_eval_envs.map_x
                    ypos = self.single_eval_envs.get_unit_by_id(agent_id).pos.y / self.single_eval_envs.map_y
                    currentPosition[agent_id] = [xpos, ypos]
                positionOfAgents.append(currentPosition)
                # Obser reward and next obs
                # eval_dones is per agent
                eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.single_eval_envs.step(eval_actions)
                # can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
                eval_actions = eval_actions.cpu().numpy()
                true_policies = true_policies.cpu().numpy()
                
                eval_obs = np.expand_dims(eval_obs, axis=0)
                eval_share_obs = np.expand_dims(eval_share_obs, axis=0)
                eval_rewards = np.expand_dims(eval_rewards, axis=0)
                eval_available_actions = np.expand_dims(eval_available_actions, axis=0)
                eval_actions = np.expand_dims(eval_actions, axis=0)
                
                true_policies = np.expand_dims(true_policies, axis=0)
                
                MyRewards.append(eval_rewards)
                MyObs.append(eval_obs)
                MyActions.append(eval_actions)
                MyPolicies.append(true_policies)

                episode_steps += 1
                
                done = np.all(eval_dones)
                
                ###
                
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
                
                env_done = np.expand_dims(done, axis=0)

                eval_rnn_states[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

                eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                eval_masks[env_done == True] = np.zeros(((env_done == True).sum(), self.num_agents, 1), dtype=np.float32)
                
                # print("Dones", done)

        data_dic = {}
        if done and 'battle_won' in eval_infos:
            battle_won = eval_infos['battle_won']
            
        # Prepare MyObs, MyActions, MyRewards into np array instead of list
        MyObs = np.stack(MyObs, axis=0)
        MyActions = np.stack(MyActions, axis=0)
        MyRewards = np.stack(MyRewards, axis=0)

        #stack positionOfAgents with dim 0 being new
        positionOfAgents = np.stack(positionOfAgents, axis=0)
            
        # Save the frames as a gif
        render_save_path = None
        if len(frames) > 0:
            # frames = self.social_influence_network.annotate_frames(frames, MyActions, MyPolicies) # This is seperate from the social influence network so it does not matter which one is used
            # Turn off policy annotation since we are doing individual evaluation
            frames = self.social_influence_network.annotate_frames(frames, MyActions, None) # This is seperate from the social influence network so it does not matter which one is used
            render_save_path = f'{self.gifs_dir}/Eval-Render-SMAC_Ep{total_num_steps}.gif'
            imageio.mimwrite(uri=render_save_path, ims=frames, format="GIF-PIL", fps=1)
            data_dic["single_ep_eval/SMAC_video"] = wandb.Video(render_save_path)
        
        

        # print("Position of Agents: ", positionOfAgents.shape)
        
        ## Do social influence evaluation here
        metrics = {}
        try:
            metrics = self.compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(MyObs, MyActions, MyRewards, total_num_steps=total_num_steps, positionOfAgents=positionOfAgents, minimal_gifs=LessGifs, frames=frames)
            
                
        except Exception as e:
            print(e)
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
        
        # if gif_save_path is not None and len(metrics.keys()) > 0:
        #     data_dic["single_ep_eval/social_influence_graph"] = wandb.Image(gif_save_path)

        if self.use_wandb:
            wandb.log(data_dic, step=total_num_steps)

        fixed_reward = data_dic.get("single_ep_eval/fixed_model_social_influence_mean_metric")
        learning_reward = data_dic.get("single_ep_eval/learning_model_social_influence_mean_metric")

        print(f"Single Ep Evaluation - Episode reward: {episode_reward}, Episode steps: {episode_steps} - Learning Social Influence Reward: {learning_reward} - Fixed Social Influence Reward: {fixed_reward}")
        
        return episode_reward, battle_won        return episode_reward, battle_won