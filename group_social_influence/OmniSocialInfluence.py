# Omni Directional Social Influence Network - needs to be for multiple agents

import numpy as np
from sklearn import metrics
import torch
import torch.distributions.kl as kl
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch.optim as optim
import glob
import networkx as nx
import imageio
import matplotlib as mpl
import os
import wandb
import datetime
from skimage.exposure import equalize_hist
import cv2

if os.environ.get("DISPLAY", "") == "":
    print("no display found. Using non-interactive Agg backend")
    mpl.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import os
import math
from .graph_analysis_helper import GraphHandler
from pathlib import Path
import matplotlib.path as mpath
import textwrap

plt.rcParams["text.usetex"] = False
# from scipy.spatial import distance

from .smac_utility.translations_4m_v_5m_to_4m import *


def convert_team_to_individual_rewards(GraphSimilarity, num_agents):
    # Reshape for team reward now is (3200)
    GraphSimilarity = np.array(GraphSimilarity)  # .reshape(-1, 1)
    # Repeat 4 times for each agent insert new axis
    GraphSimilarity = np.expand_dims(GraphSimilarity, axis=1)
    GraphSimilarity = np.expand_dims(GraphSimilarity, axis=1)
    # size is now (3200, 1, 1)
    GraphSimilarity = np.repeat(GraphSimilarity, num_agents, axis=1)
    # size is now (3200, 4, 1)

    return GraphSimilarity  # this should be one value for each timestep per agent...


def graph_pruning(graph, PercentilePrune):
    flattenGraph = graph.flatten()
    # Remove all the zeroes
    flattenGraphCleaned = flattenGraph[flattenGraph > 1e-8]

    assert (
        sum(flattenGraphCleaned) > 0
    ), "There should be some social influence in the graph"

    TwentyFifthPercentile = np.percentile(
        flattenGraphCleaned, PercentilePrune
    )  # This is per episode

    cleanedGraph = np.where(graph < TwentyFifthPercentile, 0, graph)

    return cleanedGraph


def cosine_similarity(a, b):
    # 4x4 matrix because 4 agents
    Aflat = a.flatten()
    Bflat = b.flatten()
    dot_product = np.dot(Aflat, Bflat)
    norm_a = np.linalg.norm(Aflat)
    norm_b = np.linalg.norm(Bflat)
    if norm_a == 0 or norm_b == 0:
        # print("Norm is zero")
        # print("A", a)
        # print("B", b)
        return 0  # When no SI, graph is similar but we don't want to reward this
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def cosine_similarity_for_batch(a, b, axisOfTimestep=2):
    ListByTimestepA = unstack(a, axisOfTimestep)
    ListByTimestepB = unstack(b, axisOfTimestep)
    similarities = [
        cosine_similarity(A, B) for A, B in zip(ListByTimestepA, ListByTimestepB)
    ]
    return similarities


def keep_agent_relevant_weights(agent_index, weights_array):
    """
    Keep only the specified row in the weights array, setting all other elements to zero.

    Parameters:
    agent_index (int): Index of the agent.
    weights_array (np.ndarray): 2D array of weights.

    Returns:
    np.ndarray: Modified weights array with only the specified row kept.
    """
    if not (0 <= agent_index < weights_array.shape[0]):
        raise ValueError("agent_index is out of bounds")

    # Create a one-dimensional array with the same length as the row
    row_array = np.zeros(weights_array.shape[1])

    ROW = False
    if ROW:
        # Copy the specified row
        row_array[:] = weights_array[agent_index]
    else:
        # Getting column
        row_array = weights_array[:, agent_index]

    return row_array


def individual_cosine_similarity_for_batch(a, b, axisOfTimestep=2):
    ListByTimestepA = unstack(a, axisOfTimestep)
    ListByTimestepB = unstack(b, axisOfTimestep)

    # size of the matrix is N x N while N is number of agents
    n_agents = ListByTimestepA[0].shape[0]

    assert n_agents > 1, "Number of agents should be more than 1"

    AllSimilarities = []
    for agent_index in range(n_agents):
        AgentListA = [
            keep_agent_relevant_weights(agent_index, A) for A in ListByTimestepA
        ]
        AgentListB = [
            keep_agent_relevant_weights(agent_index, B) for B in ListByTimestepB
        ]

        similarities_for_this_agent = [
            cosine_similarity(A, B) for A, B in zip(AgentListA, AgentListB)
        ]
        AllSimilarities.append(similarities_for_this_agent)

    # make AllSimilarities into a full np matric
    FullArrayOfSimilarityRewards = np.array(AllSimilarities)

    # Swap axes to get the shape (n_timestep, n_agents)
    FullArrayOfSimilarityRewards = np.swapaxes(FullArrayOfSimilarityRewards, 0, 1)

    # Add a new dimension to get the shape (n_timestep, n_agents, 1)
    FullArrayOfSimilarityRewards = np.expand_dims(FullArrayOfSimilarityRewards, axis=-1)

    return FullArrayOfSimilarityRewards


def individual_scaled_cosine_similarity_for_batch(a, b, axisOfTimestep=2):
    individual_cosine_similarities = individual_cosine_similarity_for_batch(
        a, b, axisOfTimestep
    )
    n_agents = individual_cosine_similarities.shape[1]

    SumForGroup = np.sum(individual_cosine_similarities, axis=1)
    SumForGroup = np.expand_dims(SumForGroup, axis=1)
    SumForGroup = np.repeat(SumForGroup, n_agents, axis=1)
    assert SumForGroup.shape == (
        individual_cosine_similarities.shape
    ), f"Unexpected shape for SumForGroup: {SumForGroup.shape} when individual_cosine_similarities.shape is {individual_cosine_similarities.shape}"

    # Handle division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        PercentageOfGroup = np.where(
            SumForGroup != 0, individual_cosine_similarities / SumForGroup, 0
        )
    # Log any potential issues
    if np.any(np.isnan(PercentageOfGroup)):
        print("Warning: NaN values found in PercentageOfGroup")
    if np.any(np.isinf(PercentageOfGroup)):
        print("Warning: Inf values found in PercentageOfGroup")

    group_cosine_similarity = convert_team_to_individual_rewards(
        cosine_similarity_for_batch(a, b, axisOfTimestep), n_agents
    )
    # centered_group_cosine_similarity = group_cosine_similarity - 0.5 # since group cosine similarity is between 0 and 1, centered is between -0.5 and 0.5
    agent_multiplied_group_cosine_similarity = n_agents * group_cosine_similarity
    FinalRewards = agent_multiplied_group_cosine_similarity * PercentageOfGroup

    return FinalRewards


def signal_process_remove_low_probs(probs):
    # Check the shape and type of probs
    # assert isinstance(probs, torch.Tensor), "probs should be a torch.Tensor"
    # print(f"probs shape: {probs.shape}")

    # if nd array instead of tensor
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs)

    # Remove low probabilities
    dynamicThreshold = probs.mean(dim=-1, keepdim=True) - 2 * probs.std(
        dim=-1, keepdim=True
    )

    # Check the shape and type of dynamicThreshold
    # assert isinstance(dynamicThreshold, torch.Tensor), "dynamicThreshold should be a torch.Tensor"
    # print(f"dynamicThreshold shape: {dynamicThreshold.shape}")

    probs = torch.where(probs < dynamicThreshold, torch.zeros_like(probs), probs)

    # Small epsilon value to avoid division by zero
    epsilon = 1e-8

    # Compute the sum of probabilities along the last dimension
    sum_probs = probs.sum(dim=-1, keepdim=True)

    # Add epsilon to the sum to avoid division by zero
    sum_probs = sum_probs + epsilon

    # Renormalize so all probs add up to 1
    probs = probs / sum_probs

    return probs


def softmax_to_onehot(true_policies):
    """
    Converts a np of softmax policies into one-hot vectors.

    Args:
        true_policies: A tensor of shape (batchsize, agent_number, num_classes)
        containing softmax probabilities.

    Returns:
        A np of shape (batchsize, agent_number, num_classes) where the 1 is
        at the argmax of each softmax policy vector.
    """

    # Ensure the input np has the expected shape
    if len(true_policies.shape) != 3:
        raise ValueError(
            "Input tensor must have shape (batchsize, agent_number, num_classes)"
        )

    # Find the argmax (index of the maximum value) along the last dimension (class axis)
    argmax_indices = np.argmax(true_policies, axis=-1)  # -1 for last dimension

    # Convert to one-hot vectors
    one_hot_vectors = np.eye(true_policies.shape[-1])[argmax_indices]

    # print("true_policies", true_policies.shape)
    # print("one_hot_vectors", one_hot_vectors.shape)

    return one_hot_vectors


def getMarkers(ListOfThings):
    # ListOfMarkers = []
    for action_item in ListOfThings:

        if action_item == "Joint Focus Fire" or action_item == "Team Focus Fire":
            star = mpath.Path.unit_regular_star(6)
            return star
        elif action_item == "Pause":
            return "s"
        elif action_item == "Mark":
            # ListOfMarkers.append("X") # filled x
            return "X"
        elif action_item == "Failed Mark":
            # ListOfMarkers.append("x")
            return "x"
        elif action_item == "Scattered Fire":
            star = mpath.Path.unit_regular_star(6)
            circle = mpath.Path.unit_circle()
            # concatenate the circle with an internal cutout of the star
            cut_star = mpath.Path(
                vertices=np.concatenate([circle.vertices, star.vertices[::-1, ...]]),
                codes=np.concatenate([circle.codes, star.codes]),
            )
            return cut_star
            # ListOfMarkers.append(cut_star)
            # ListOfMarkers.append("o")
        elif action_item == "1U Move":  # use
            # ListOfMarkers.append("^")
            return "^"
        elif action_item == "Joint Move":
            # ListOfMarkers.append(">")
            return ">"
        elif action_item == "Team Move":
            # ListOfMarkers.append(">")
            return ">"
        elif action_item == "Partial Pause":
            # ListOfMarkers.append("s")
            return "s"
        elif action_item == "Part Focus Fire":
            return "."
        elif action_item == "1U Shooting":
            # ListOfMarkers.append(".")
            return "."
        else:
            # ListOfMarkers.append(".")
            return "."

    # return ' '.join(ListOfMarkers)


def geo_mean_overflow(iterable):
    # if there is a zero, then the geometric mean is zero
    if 0 in iterable:
        return 0
    return np.exp(np.log(iterable).mean())


def convert_joint_to_individual(joint_obs):
    # Convert joint_obs to a numpy array
    joint_obs_array = np.array(joint_obs)

    # Swap the first two dimensions
    # print(joint_obs_array.shape)
    # print(joint_obs_array.shape)
    if len(joint_obs_array.shape) == 5:
        joint_obs_swapped = joint_obs_array.transpose(1, 0, 2, 3, 4)
    else:
        joint_obs_swapped = joint_obs_array.transpose(1, 0, *range(2, joint_obs_array.ndim))
    # joint_obs_swapped = joint_obs_array.transpose(1, 0, 4,2,3)
    # print(joint_obs_swapped.shape)
    # Convert joint_obs_swapped to a tensor
    joint_obs_tensor = torch.from_numpy(joint_obs_swapped)
    return joint_obs_tensor


def convert_individual_to_joint(individual_obs):
    # Convert individual_obs to a numpy array
    individual_obs_array = np.array(individual_obs)

    # Swap the first two dimensions
    individual_obs_swapped = individual_obs_array.transpose(
        1, 0, *range(2, individual_obs_array.ndim)
    )
    # Convert individual_obs_swapped to a tensor
    individual_obs_tensor = torch.from_numpy(individual_obs_swapped)
    return individual_obs_tensor


# Function to binary encode numbers
def binary_encode(numbers, num_digits):
    # Check if numbers is a single number or a list/array
    if np.isscalar(numbers):
        numbers = [numbers]

    # Convert the numbers to binary strings and then to a numpy array of integers
    binary = np.array(
        [list(np.binary_repr(number, num_digits)) for number in numbers], dtype=int
    )

    return binary


def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)


def kl_divergence(predMu, predStd, trueMu, trueStd):
    # This is KL Loss https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians # p is predicted, q is true, 1 = Pred, 2 = True
    singleLoss = (
        torch.log(trueStd / predStd)
        + ((predStd.pow(2) + (predMu - trueMu).pow(2)) / (2 * trueStd.pow(2)))
        - 0.5
    )
    return singleLoss


# def jensen_shannon_divergence(p, q):
#     p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
#     m = (0.5 * (p + q)).log()
#     return 0.5 * (kl.kl_divergence(m, p.log()) + kl.kl_divergence(m, q.log()))


class SocialInfluenceManager:

    def __init__(self, **kwargs):
        # super(CLASS_NAME, self).__init__(*args, **kwargs)
        self.num_agents = kwargs["num_agents"]
        self.hidden_size = kwargs["hidden_size"]
        self.device = kwargs["device"]
        self.num_env_steps = kwargs["num_env_steps"]
        self.action_space = kwargs["action_space"]
        self.n_obs = kwargs["n_obs"]
        self.episode_length = 5000  # Not used practically

        self.gifs_dir = kwargs["gifs_dir"]
        self.model_dir_si_fixed = kwargs["model_dir_si_fixed"]
        self.model_dir_si_learning = kwargs["model_dir_si_learning"]
        self.compare_si_policy = kwargs["compare_si_policy"]

        self.run_mixed_population = kwargs.get("run_mixed_population", False)
        self.si_loss_type = kwargs["si_loss_type"]
        self.only_use_argmax_policy = kwargs["only_use_argmax_policy"]

        self.ShortHandActions = kwargs.get("ShortHandActions", [])

        self.TwoDifferentTeamEnvs = False  # TODO No different envs in pursuit

        if not os.path.exists(self.gifs_dir):
            os.makedirs(self.gifs_dir)

        if self.compare_si_policy:
            first_gif_dir = os.path.join(self.gifs_dir, "DualSIComparison-Fixed")
            if not os.path.exists(first_gif_dir):
                os.makedirs(first_gif_dir)

            second_gif_dir = os.path.join(self.gifs_dir, "DualSIComparison-Learning")
            if not os.path.exists(second_gif_dir):
                os.makedirs(second_gif_dir)

        if self.run_mixed_population:
            first_gif_dir = os.path.join(self.gifs_dir, "MixedPopulation_SI_TeamA")
            if not os.path.exists(first_gif_dir):
                os.makedirs(first_gif_dir)

            second_gif_dir = os.path.join(self.gifs_dir, "MixedPopulation_SI_TeamB")
            if not os.path.exists(second_gif_dir):
                os.makedirs(second_gif_dir)

        # self.gifs_dir = str(kwargs["run_dir"] / 'Gifs')
        # if not os.path.exists(self.gifs_dir):
        #     os.makedirs(self.gifs_dir)

        if kwargs["discrete_actions"]:
            is_discrete_actions = True
            action_dim = 1
            self.social_influence_n_counterfactuals = self.action_space - 2
            print(
                "Setting social influence counterfactuals to action space size since discrete actions"
            )
        else:
            is_discrete_actions = False
            action_dim = self.action_space
            self.social_influence_n_counterfactuals = kwargs[
                "social_influence_n_counterfactuals"
            ]  # usually 10

        # if kwargs["policy"] is a model
        # self.policy = kwargs["policy"]

        # if isinstance(kwargs["policy"], torch.nn.Module):
        #     self.policy = kwargs["policy"]
        # else:
        #     # if kwargs["policy"] is a TensorDictSequential

        params = {
            "num_agents": self.num_agents,
            "action_space": self.action_space,
            "action_dim": action_dim,  # because discrete action for SMAC
            "discrete_actions": is_discrete_actions,
            "n_obs": self.n_obs,
            "n_hidden_units": self.hidden_size,
            "device": self.device,
            "num_env_steps": self.num_env_steps,
            "gifs_dir": self.gifs_dir,
            "social_influence_n_counterfactuals": self.social_influence_n_counterfactuals,
            "si_loss_type": self.si_loss_type,
            "only_use_argmax_policy": self.only_use_argmax_policy,
        }

        if params["discrete_actions"]:
            params["social_influence_n_counterfactuals"] = (
                self.action_space - 1
            )  # TODO  # because removing the agent's own action and the noops action 0
            self.social_influence_n_counterfactuals = params[
                "social_influence_n_counterfactuals"
            ]

        MyInfo = params.copy()
        if self.compare_si_policy:
            learning_net_params = MyInfo.copy()
            learning_net_params.update({"gifs_dir": second_gif_dir})
            fixed_net_params = MyInfo.copy()
            fixed_net_params.update({"gifs_dir": first_gif_dir})
            self.social_influence_net = GroupSocialInfluence(learning_net_params)
            self.social_influence_net_fixed = GroupSocialInfluence(fixed_net_params)

            if kwargs["model_dir_si_fixed"] is not None:
                self.restore_social_influence(
                    self.social_influence_net_fixed, kwargs["model_dir_si_fixed"]
                )
            else:
                print("WARNING: No model dir for social influence fixed")

            if kwargs["model_dir_si_learning"] is not None:
                self.restore_social_influence(
                    self.social_influence_net, kwargs["model_dir_si_learning"]
                )
            else:
                print("WARNING:No model dir for social influence learning")
        elif self.run_mixed_population:

            A_net_params = MyInfo.copy()
            A_net_params.update({"gifs_dir": first_gif_dir})
            A_net_params["TwoDifferentTeamEnvs_TeamBAdjustments"] = False

            self.social_influence_net_A = GroupSocialInfluence(A_net_params)

            B_net_params = MyInfo.copy()
            B_net_params.update({"gifs_dir": second_gif_dir})
            B_net_params["TwoDifferentTeamEnvs_TeamAAdjustments"] = False

            self.social_influence_net_B = GroupSocialInfluence(B_net_params)

            if kwargs["model_dir_si_fixed"] is not None:
                self.restore_social_influence(
                    self.social_influence_net_A, kwargs["model_dir_si_fixed"]
                )
            else:
                print("WARNING: No model dir for social influence fixed")

            if kwargs["model_dir_si_learning"] is not None:
                self.restore_social_influence(
                    self.social_influence_net_B, kwargs["model_dir_si_learning"]
                )
            else:
                print(
                    "WARNING:No model dir for the team social influences - not a problem if doing a run with untrained team b"
                )

        else:
            self.social_influence_net = GroupSocialInfluence(params)
            if kwargs["model_dir_si_learning"] is not None:
                self.restore_social_influence(
                    self.social_influence_net, kwargs["model_dir_si_learning"]
                )
                print(
                    "Restored social influence learning model for this load run as it was provided."
                )

        self.graph_node_positions = None

        ### Managing graphs
        self.si_graph_checker = GraphHandler(self.num_agents, self.gifs_dir)

        #### constant for SMAC  #TODO what to do?
        self.actionTypes = [
            "Team Focus Fire",
            "Joint Focus Fire",
            "Part Focus Fire",
            "Scattered Fire",
            "Team Move",
            "Joint Move",
            "1U Move",
            "Mark",
            "Failed Mark",
            "Pause",
            "Partial Pause",
            "1U Shooting",
        ]
        self.CentralMemory = {"internalCounter": 0}

    def clear_gifs(self):
        # remove all the gifs in self.gifs_dir folder and sub folders
        for gif_file in glob.glob(self.gifs_dir + "**/*.gif", recursive=True):
            try:
                os.remove(gif_file)
            except OSError as e:
                print("Error: %s : %s" % (gif_file, e.strerror))

    def get_loss(
        self, IndividualObs, IndividualActions, next_policies, AGENTS_CAN_DIE=True
    ):
        """social_influence_loss"""
        IndividualActions_B = IndividualActions
        IndividualObs_B= IndividualObs
        next_policies_B = next_policies
        
        if self.run_mixed_population:
            ALoss = self.social_influence_net_A.get_loss(
                IndividualObs,
                IndividualActions,
                next_policies,
                AGENTS_CAN_DIE=AGENTS_CAN_DIE,
            )

            BLoss = self.social_influence_net_B.get_loss(
                IndividualObs_B,
                IndividualActions_B,
                next_policies_B,
                AGENTS_CAN_DIE=AGENTS_CAN_DIE,
            )
            metrics = {
                "Social Influence Loss (Team A)": ALoss,
                "Social Influence Loss (Team B)": BLoss,
            }

        else:

            alongside = self.social_influence_net.get_loss(
                IndividualObs,
                IndividualActions,
                next_policies,
                AGENTS_CAN_DIE=AGENTS_CAN_DIE,
            )
            metrics = {"Social Influence Loss (Alongside Policy)": alongside}
            if self.compare_si_policy:
                final_fixed = self.social_influence_net_fixed.get_loss(
                    IndividualObs,
                    IndividualActions,
                    next_policies,
                    AGENTS_CAN_DIE=AGENTS_CAN_DIE,
                )
                metrics["Social Influence Loss (Final Fixed Policy)"] = final_fixed
        return metrics

    def restore_social_influence(self, model, model_dir):
        model.load(model_dir)

    def iso_similarity_for_batch(self, a, b, axisOfTimestep=2):
        ListByTimestepA = unstack(a, axisOfTimestep)
        ListByTimestepB = unstack(b, axisOfTimestep)
        ListTSGraphA = [nx.from_numpy_matrix(A) for A in ListByTimestepA]
        ListTSGraphB = [nx.from_numpy_matrix(B) for B in ListByTimestepB]
        similarities = [
            nx.is_isomorphic(A, B) for A, B in zip(ListTSGraphA, ListTSGraphB)
        ]

        # convert true to 1 and false to 0
        similarities = [int(similarity) for similarity in similarities]

        return similarities

    def edge_similarity_for_batch(self, a, b, axisOfTimestep=2):
        ListByTimestepA = unstack(a, axisOfTimestep)
        ListByTimestepB = unstack(b, axisOfTimestep)

        ListTSGraphA = [nx.from_numpy_matrix(A) for A in ListByTimestepA]
        ListTSGraphB = [nx.from_numpy_matrix(B) for B in ListByTimestepB]

        NumberOfTotalPossibleEdges = self.si_graph_checker.numberEdgesPossible
        similarities = [
            nx.number_of_edges(A) - nx.number_of_edges(B) / NumberOfTotalPossibleEdges
            for A, B in zip(ListTSGraphA, ListTSGraphB)
        ]
        return similarities

    def more_efficient_compute_group_social_influence_rewards_all_types(
        self, obs, actions, TeamBObs=None, chosen_pruning_percentile=35, episode=None
    ):
        """Calculate group social influence rewards for the collected data and returns it. Also saves a social influence gif to gif directory."""
        ## Calculate the social influence rewards
        if self.run_mixed_population:
            AMetrics = self.social_influence_net_A.calc_social_influence_reward_group(
                obs, actions
            )

            if TeamBObs is not None:
                BMetrics = (
                    self.social_influence_net_B.calc_social_influence_reward_group(
                        TeamBObs, actions
                    )
                )
            else:
                raise ValueError(
                    "Team B observations are required for mixed population calc in compute_group_social_influence_rewards"
                )

            perAgentSocialInfluenceRewardsA, graphA = AMetrics[0]
            perAgentSocialInfluenceRewardsB, graphB = BMetrics[0]

            NumberOfTimesteps = obs.shape[1]  # 4, 3200, 88

            MetricDict = {}

            ### No Pruning
            NoPrune_Cosine_GraphSimilarity = cosine_similarity_for_batch(
                graphA, graphB, axisOfTimestep=2
            )  # returns a list of rewards for each timestep
            MetricDict["graph_similarity-Cosine-NoPruning"] = (
                NoPrune_Cosine_GraphSimilarity
            )

            assert (
                len(NoPrune_Cosine_GraphSimilarity) == NumberOfTimesteps
            ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
                len(NoPrune_Cosine_GraphSimilarity), obs.shape
            )

            PercentilePrune = [chosen_pruning_percentile]

            for PercentPruning in PercentilePrune:
                graphA = graph_pruning(graphA, PercentPruning)
                graphB = graph_pruning(graphB, PercentPruning)

                Name = f"-Pruning_{PercentPruning}"

                ####################### Cosine
                NoPrune_Cosine_GraphSimilarity = cosine_similarity_for_batch(
                    graphA, graphB, axisOfTimestep=2
                )  # returns a list of rewards for each timestep
                EntryID = f"graph_similarity-Cosine{Name}"
                MetricDict[EntryID] = NoPrune_Cosine_GraphSimilarity

                assert (
                    len(NoPrune_Cosine_GraphSimilarity) == NumberOfTimesteps
                ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
                    len(NoPrune_Cosine_GraphSimilarity), obs.shape
                )

                ####################### GraphSimilarity - Same isometric graph
                if True:  # Do we want to print the individual agents similarities
                    GraphSimilarity = individual_scaled_cosine_similarity_for_batch(
                        graphA, graphB, axisOfTimestep=2
                    )
                    # timestep, agents, reward
                    ListByAgents = unstack(GraphSimilarity, 1)

                    for agent in range(self.num_agents):
                        AgentEntryID = EntryID + f"Agent-{agent}"
                        MetricDict[AgentEntryID] = ListByAgents[agent]

                ####################### GraphSimilarity - Same isometric graph
                if False:  # Can use but we don't need these metrics
                    SameGraph_GraphSimilarity = self.iso_similarity_for_batch(
                        graphA, graphB, axisOfTimestep=2
                    )  # returns a list of rewards for each timestep
                    EntryID = f"graph_similarity-Iso{Name}"
                    MetricDict[EntryID] = SameGraph_GraphSimilarity

                    assert (
                        len(SameGraph_GraphSimilarity) == NumberOfTimesteps
                    ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
                        len(SameGraph_GraphSimilarity), obs.shape
                    )

                    ####################### GraphSimilarity - Same isometric graph
                    NumberEdges_GraphSimilarity = self.edge_similarity_for_batch(
                        graphA, graphB, axisOfTimestep=2
                    )  # returns a list of rewards for each timestep
                    EntryID = f"graph_similarity-CountEdgeDifference{Name}"
                    MetricDict[EntryID] = SameGraph_GraphSimilarity

                    assert (
                        len(SameGraph_GraphSimilarity) == NumberOfTimesteps
                    ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
                        len(SameGraph_GraphSimilarity), obs.shape
                    )

                action_analysis = self.smac_action_analysis(actions)

                NewMetricsDict = {}
                for k, v in MetricDict.items():
                    assert (
                        len(v) == NumberOfTimesteps
                    ), f"Graph similarity should be one value for each timestep, instead {len(v)} values were returned for {obs.shape}"

                    # newV = self.convert_team_to_individual_rewards(v) # Don't need to because I am not using like that
                    # NewMetricsDict[k] = newV
                    MeanValue = np.mean(v)
                    NewMetricsDict[k + " (Ep Mean)"] = MeanValue

                    # PlotChangeOverTime
                    ImgPath = self.plotTimestepGraphRewards(
                        rewards=v,
                        action_analysis=action_analysis,
                        episode=episode,
                        extra_string=k.replace(
                            "graph_similarity-", "Graph Similarity: "
                        ),
                    )
                    NewMetricsDict[k + " (Over Ep)"] = wandb.Image(ImgPath)

                return NewMetricsDict

        else:
            metrics = self.social_influence_net.calc_social_influence_reward_group(
                obs, actions
            )
            perAgentSocialInfluenceRewards, graph = metrics[0]

        return perAgentSocialInfluenceRewards

    # def convert_team_to_individual_rewards(self, GraphSimilarity):
    #     # Reshape for team reward now is (3200)
    #     GraphSimilarity = np.array(GraphSimilarity)  # .reshape(-1, 1)
    #     # Repeat 4 times for each agent insert new axis
    #     GraphSimilarity = np.expand_dims(GraphSimilarity, axis=1)
    #     GraphSimilarity = np.expand_dims(GraphSimilarity, axis=1)
    #     # size is now (3200, 1, 1)
    #     GraphSimilarity = np.repeat(GraphSimilarity, self.num_agents, axis=2)
    #     # size is now (3200, 4, 1)

    #     return (
    #         GraphSimilarity  # this should be one value for each timestep per agent...
    #     )
    def compute_individual_group_social_influence_rewards(
        self,
        obs,
        actions,
        TeamBObs=None,
        chosenRewardType="cosine_similarity",
        chosen_pruning_percentile=25,
    ):
        """Calculate group social influence rewards for the collected data and returns it. Also saves a social influence gif to gif directory."""
        ## Calculate the social influence rewards
        if self.run_mixed_population:
            AMetrics = self.social_influence_net_A.calc_social_influence_reward_group(
                obs, actions
            )

            if TeamBObs is not None:
                BMetrics = (
                    self.social_influence_net_B.calc_social_influence_reward_group(
                        TeamBObs, actions
                    )
                )
            else:
                raise ValueError(
                    "Team B observations are required for mixed population calc in compute_group_social_influence_rewards"
                )

            perAgentSocialInfluenceRewardsA, graphA = AMetrics[0]
            perAgentSocialInfluenceRewardsB, graphB = BMetrics[0]

            if chosen_pruning_percentile > 0:
                graphA = graph_pruning(graphA, chosen_pruning_percentile)
                graphB = graph_pruning(graphB, chosen_pruning_percentile)

            if chosenRewardType == "Cosine":
                # GraphSimilarity - Cosine Similarity
                GraphSimilarity = individual_scaled_cosine_similarity_for_batch(
                    graphA, graphB, axisOfTimestep=2
                )  # returns a list of rewards for each timestep

            if chosenRewardType == "SameGraph" or chosenRewardType == "EdgeDifference":
                raise NotImplementedError(
                    "Not doing other reward types for individual rewards"
                )
            NumberOfTimesteps = obs.shape[1]  # 4, 3200, 88
            # assert (
            #     len(GraphSimilarity) == NumberOfTimesteps
            # ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
            #     len(GraphSimilarity), obs.shape
            # )

            # return self.convert_team_to_individual_rewards(GraphSimilarity)

            # Confirm that shape of GraphSimilarity is 3200, 4, 1
            assert (
                GraphSimilarity.shape[0] == NumberOfTimesteps
            ), "Graph similarity should get first axis = num timestep"
            assert (
                len(GraphSimilarity.shape) == 3
            ), f"Graph similarity should have length 3 but instead we have {len(GraphSimilarity)}"
            return GraphSimilarity

        else:
            raise NotImplementedError(
                "Should not calc individual SI influence for not mixed"
            )
            metrics = self.social_influence_net.calc_social_influence_reward_group(
                obs, actions
            )
            perAgentSocialInfluenceRewards, graph = metrics[0]

        return perAgentSocialInfluenceRewards
    def compute_group_social_influence_rewards(
        self,
        obs,
        actions,
        TeamBObs=None,
        chosenRewardType="cosine_similarity",
        chosen_pruning_percentile=25,
    ):
        """Calculate group social influence rewards for the collected data and returns it. Also saves a social influence gif to gif directory."""
        ## Calculate the social influence rewards
        if self.run_mixed_population:
            AMetrics = self.social_influence_net_A.calc_social_influence_reward_group(
                obs, actions
            )

            if TeamBObs is not None:
                BMetrics = (
                    self.social_influence_net_B.calc_social_influence_reward_group(
                        TeamBObs, actions
                    )
                )
            else:
                raise ValueError(
                    "Team B observations are required for mixed population calc in compute_group_social_influence_rewards"
                )

            perAgentSocialInfluenceRewardsA, graphA = AMetrics[0]
            perAgentSocialInfluenceRewardsB, graphB = BMetrics[0]

            if chosen_pruning_percentile > 0:
                graphA = graph_pruning(graphA, chosen_pruning_percentile)
                graphB = graph_pruning(graphB, chosen_pruning_percentile)

            if chosenRewardType == "Cosine":
                # GraphSimilarity - Cosine Similarity
                # TODO try different types of graph similarity measures here e.g. number of edges
                GraphSimilarity = cosine_similarity_for_batch(
                    graphA, graphB, axisOfTimestep=2
                )  # returns a list of rewards for each timestep

            if chosenRewardType == "SameGraph":
                # GraphSimilarity - Same isometric graph
                GraphSimilarity = self.iso_similarity_for_batch(
                    graphA, graphB, axisOfTimestep=2
                )  # returns a list of rewards for each timestep

            if chosenRewardType == "EdgeDifference":
                # GraphSimilarity - Same isometric graph
                GraphSimilarity = self.edge_similarity_for_batch(
                    graphA, graphB, axisOfTimestep=2
                )

            NumberOfTimesteps = obs.shape[1]  # 4, 3200, 88
            assert (
                len(GraphSimilarity) == NumberOfTimesteps
            ), "Graph similarity should be one value for each timestep, instead {} values were returned for {} obs shape".format(
                len(GraphSimilarity), obs.shape
            )

            return convert_team_to_individual_rewards(GraphSimilarity,self.num_agents)

        else:
            metrics = self.social_influence_net.calc_social_influence_reward_group(
                obs, actions
            )
            perAgentSocialInfluenceRewards, graph = metrics[0]

        return perAgentSocialInfluenceRewards

    def _action_checker_shooting(self, single_action_set):
        """Check if all agents are shooting"""
        # - 6 = (agent 1 if they are in range)
        # - 7 = (agent 2 if they are in range)
        # - 8 = (agent 3 if they are in range)
        cleaned_single_action_set = single_action_set[
            single_action_set > 0
        ]  # Who is alive
        numberAgentsAlive = len(cleaned_single_action_set)
        numberOfAgentsShooting = np.sum(cleaned_single_action_set >= 6)
        AllLivingAgentsShooting = np.all(cleaned_single_action_set >= 6)

        whoIsShooting = cleaned_single_action_set[cleaned_single_action_set >= 6]
        numberAgentsShooting = len(whoIsShooting)

        if numberAgentsShooting == 0:
            return False  # no one is shooting
        elif numberAgentsShooting == 1:
            return "1U Shooting"
        elif numberAgentsShooting > 1:  # Just to be sure
            # More than one agent is shooting
            values, counts = np.unique(whoIsShooting, return_counts=True)
            NumberOfAgentsOnSameTarget = max(counts)
            allAgentsShootingAreOnSameTarget = (
                NumberOfAgentsOnSameTarget == numberOfAgentsShooting
            )
            if (
                allAgentsShootingAreOnSameTarget
                and AllLivingAgentsShooting
                and NumberOfAgentsOnSameTarget > 1
            ):
                return "Team Focus Fire"  # When all agents are shooting the same target
            elif allAgentsShootingAreOnSameTarget and NumberOfAgentsOnSameTarget == 2:
                return "Joint Focus Fire"  # when more than one agent is shooting the same target
            elif NumberOfAgentsOnSameTarget > 1:
                return "Part Focus Fire"  # when more than one agent is shooting the same target but not all agents are shooting
            elif (
                len(values) > 1
            ):  # if there are multiple agents shooting different targets
                return "Scattered Fire"

        return False  # no agent shooting

    def _action_checker_movement(self, single_action_set):
        """Check if all agents are moving"""
        # - 2 = North
        # - 3 = South
        # - 4  = East
        # - 5 = West
        cleaned_single_action_set = single_action_set[
            single_action_set > 0
        ]  # Who is alive
        numberAgentsAlive = len(cleaned_single_action_set)
        AgentsThatAreMoving = cleaned_single_action_set[cleaned_single_action_set >= 2]
        AgentsThatAreMoving = AgentsThatAreMoving[AgentsThatAreMoving <= 5]
        NumberAgentsMoving = len(AgentsThatAreMoving)

        if NumberAgentsMoving == 0:
            return False  # No agents moving
        elif NumberAgentsMoving == 1:
            return "1U Move"
        elif NumberAgentsMoving == numberAgentsAlive:
            return "Team Move"
        elif NumberAgentsMoving < numberAgentsAlive:
            return "Joint Move"

        return False

    def _action_checker_pause(self, single_action_set):
        """Check if all agents are pausing"""
        cleaned_single_action_set = single_action_set[single_action_set > 0]
        numberAgentsAlive = len(cleaned_single_action_set)
        AgentsPausing = cleaned_single_action_set[cleaned_single_action_set == 1]
        numberAgentsPausing = len(AgentsPausing)
        AllLivingAgentsPausing = numberAgentsPausing == numberAgentsAlive
        if AllLivingAgentsPausing:
            return "Pause"
        elif numberAgentsPausing > 0:
            return "Partial Pause"
        else:
            return False

    def annotate_frames(self, frames, actions, policies=None):
        """add a caption of the joint string in action for each frame"""
        actionAnalysis = self.smac_action_analysis(actions)
        if policies is not None:
            policyAnalysis = self.policy_analysis(policies, actions)
            assert len(policyAnalysis.keys()) == len(
                actionAnalysis.keys()
            ), "Policies and actions should be the same length"
        NewFrameSet = []
        height, _, _ = frames[0].shape
        org = (120, round(height * 4 // 5) + 20)
        actions = actions.squeeze()
        for i, frame in enumerate(frames):
            frameCount = i - 1
            if frameCount < 0:
                Nframe = cv2.putText(
                    img=frame,
                    text="Start",
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                NewFrameSet.append(Nframe)
            elif i >= len(actionAnalysis.keys()) - 1:
                # This is the last frame
                Nframe = cv2.putText(
                    img=frame,
                    text="End",
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                NewFrameSet.append(Nframe)
            else:
                actionBlurb = actionAnalysis.get(frameCount, [])
                Nframe = frame
                #### Action Annotation
                if len(actionBlurb) > 0:
                    if len(actionBlurb) == 1:
                        stringOfJoinedText = str(actionBlurb[0])
                        Nframe = cv2.putText(
                            img=frame,
                            text=stringOfJoinedText,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.3,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                    else:
                        # stringOfJoinedText = "\n".join(item for item in actionBlurb if isinstance(item, str))
                        for count, info in enumerate(actionBlurb):
                            LevelOrg = (org[0], org[1] + (50 * count))
                            Nframe = cv2.putText(
                                img=frame,
                                text=info,
                                org=LevelOrg,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.3,
                                color=(255, 255, 255),
                                thickness=2,
                                lineType=cv2.LINE_AA,
                            )

                #### policyAnalysis Annotation
                if policies is not None:
                    policyDiagrams = policyAnalysis.get(
                        frameCount, {}
                    )  # should always be a dictionary
                    # policyDiagrams is a dictionary of the form {agent number: policy}

                    ## TODO draw the diagram on the frame like a matplotlib plot, then convert to cv2 image, you need NAgents subplots for each policy and one big one for the current frame, annotate on top
                    OriginalFrame = Nframe
                    # fig, ax = plt.subplots(1, 1)
                    # ax.imshow(OriginalFrame)

                    # [left, bottom, width, height]
                    fig = plt.figure(figsize=(10, 10))
                    OriginalAx = fig.add_axes([0.05, 0.05, 0.55, 0.95], zorder=1)

                    AllPolicyGraphAx = fig.add_axes([0.65, 0.05, 0.30, 0.95], zorder=2)
                    AllPolicyGraphAx.set_facecolor((1, 1, 1, 0.7))
                    AllPolicyGraphAx.axis("off")
                    AllPolicyGraphAx.set_title("Policy Distribution")

                    # NumberOfAgents to plot
                    FractionOfSpaceUsed = 0.9
                    SpacePerGraph = FractionOfSpaceUsed / self.num_agents

                    OriginalAx.imshow(OriginalFrame)

                    # Do the policy analysis
                    for agent in range(self.num_agents):
                        myTuple = policyDiagrams.get(agent, None)
                        if myTuple is not None:
                            current_policy, action = myTuple
                            # Draw the policy on the frame
                            addSubplot = fig.add_axes(
                                [
                                    0.75,
                                    0.05 + (SpacePerGraph * agent),
                                    0.25,
                                    SpacePerGraph - 0.05,
                                ],
                                zorder=3,
                            )
                            addSubplot.set_facecolor((1, 1, 1, 0.8))

                            plotSuperLong = agent == 0  # only first agent at bottom
                            addSubplot = self.plotPolicyIntoHistogram(
                                addSubplot,
                                policy=current_policy,
                                action=action,
                                agent_id=agent,
                                plotXLabelsLong=plotSuperLong,
                            )
                    # fig.tight_layout()
                    fig.canvas.draw()

                    # convert canvas to image
                    policyImage = np.frombuffer(
                        fig.canvas.tostring_rgb(), dtype=np.uint8
                    )
                    policyImage = policyImage.reshape(
                        fig.canvas.get_width_height()[::-1] + (3,)
                    )
                    # policyImage = cv2.cvtColor(policyImage, cv2.COLOR_RGB2BGR)
                    Nframe = policyImage

                # Temporarily print the action values
                actionValues = actions[frameCount]

                actionString = ",".join(
                    self.actionTranslater(item) for item in actionValues
                )
                Nframe = cv2.putText(
                    img=Nframe,
                    text=actionString,
                    org=(org[0], org[1] + (50 * 3)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

                NewFrameSet.append(Nframe)

                if policies is not None:
                    try:
                        plt.close(fig)
                        plt.cla()
                    except Exception as e:
                        print(e)

        return NewFrameSet

    def smac_action_analysis(self, actions):
        """Given the actions, return the group name for joint action taken
        Since getting the actions directly, it is not reshaped for us
        """
        # if actions is a list # if it is a torch tensor
        if isinstance(actions, list) or isinstance(actions, torch.Tensor):
            actions = (
                convert_individual_to_joint(actions).cpu().numpy()
            )  # comes from prepped data
            actions = actions.squeeze()

        if len(actions.shape) == 4:  # Actions (18, 1, 3, 1)
            actions = actions.squeeze()  # come from game

        # print("Actions", actions.shape) # 18, 3

        ## Plot the basic actions on a graph
        # print("Actions", actions) # shape is batchsize, agents
        ListByTimestep = unstack(
            actions, 0
        )  # (batchsize, 3) where 3 is the number of agents, batchsize is the number of timesteps

        # - 0 = No ops (agent is dead)
        # - 1 = Stop
        # - 2 = North
        # - 3 = South
        # - 4  = East
        # - 5 = West
        # - 6 = (agent 1 if they are in range)
        # - 7 = (agent 2 if they are in range)
        # - 8 = (agent 3 if they are in range)

        Labels = {}
        for i, current_action_set in enumerate(ListByTimestep):
            Labels[i] = []
            ### Check for Shooting
            # If all agents are shooting
            ShootingLabel = self._action_checker_shooting(current_action_set)
            if ShootingLabel:
                Labels[i].append(ShootingLabel)

            MovementLabel = self._action_checker_movement(current_action_set)
            if MovementLabel:
                Labels[i].append(MovementLabel)

            PauseLabel = self._action_checker_pause(current_action_set)
            if PauseLabel:
                Labels[i].append(PauseLabel)

            if PauseLabel == "Partial Pause" and ShootingLabel == "1U Shooting":
                Labels[i].append("Mark")
                Labels[i].remove("Partial Pause")
                Labels[i].remove("1U Shooting")

            if PauseLabel == "Partial Pause" and ShootingLabel == "Joint Focus Fire":
                Labels[i].append("Mark")
                Labels[i].remove("Partial Pause")
                Labels[i].remove("Joint Focus Fire")

            if PauseLabel == "Partial Pause" and ShootingLabel == "Part Focus Fire":
                Labels[i].append("Mark")
                Labels[i].remove("Partial Pause")
                Labels[i].remove("Part Focus Fire")

            if PauseLabel == "Partial Pause" and ShootingLabel == "Scattered Fire":
                Labels[i].append("Failed Mark")
                Labels[i].remove("Partial Pause")
                Labels[i].remove("Scattered Fire")

        return Labels

    def compute_group_social_influence_rewards_evaluation_single_episode(
        self,
        obs,
        actions,
        total_num_steps=0,
        positionOfAgents=None,
        minimal_gifs=True,
        graph_gifs=False,
        frames=None,
        policies=None,
        whichNetToUse=None,
        pruning_amount_for_group_similarity=35):
        
        if whichNetToUse is None:
            MyNet = self.social_influence_net
        else:
            MyNet = whichNetToUse

        si_grp = MyNet.calc_social_influence_reward_group(obs, actions)
        perAgentSocialInfluenceRewards, graph = si_grp[0]
        _, variance_graph = si_grp[1]

        reformat_actions = convert_individual_to_joint(actions)
        aliveMask = np.ones_like(reformat_actions)
        # if action = 0 then agent is dead
        # aliveMask[reformat_actions == 0] = 0 #TODO All agents are alive throughout the game

        action_analysis = self.smac_action_analysis(actions)

        SAVEINFO = (
            total_num_steps <= 9600 or total_num_steps % 1000000 < 3200
        )  # SO that we don't save too many graphs just for testing purposes
        if SAVEINFO:
            save_dir = self.gifs_dir.replace("Gifs", "GraphData")

            Path(save_dir).mkdir(parents=True, exist_ok=True)

            # save graph to numpy file
            np.save(f"{save_dir}/social_influence_graph_{total_num_steps}.npy", graph)

            np.save(f"{save_dir}/alive_mask_{total_num_steps}.npy", aliveMask)

            np.save(
                f"{save_dir}/positionOfAgents_{total_num_steps}.npy", positionOfAgents
            )

        # print("aliveMask shape", aliveMask.shape)

        # last two
        data_dic = {}
        gif_save_path = None
        secondary_pruned_gif_save_path = None

        learning_analysis = self.SI_analysis(graph, aliveMask, total_num_steps)
        data_dic.update(learning_analysis)

        # if minimal_gifs:
        #     CareAbout = [("_metric", graph)]
        # else:
        #     CareAbout = [("_metric", graph), ("_variance", variance_graph)]

        CareAbout = [("_metric", graph)]

        # DataForSICalculation = (obs, actions)

        if policies is None:
            BigInfoOfSIPolicies = None
        else:
            CounterFactualInfo = (
                MyNet.calc_social_influence_reward_with_counterfactuals_returned(
                    obs, actions
                )
            )
            BigInfoOfSIPolicies = (CounterFactualInfo, actions, policies)
            # print("compute GSIR actions", actions.shape) compute GSIR actions torch.Size([3, 18, 1])
            # print("compute GSIR policies", policies.shape) compute GSIR policies (3, 18, 9)

        for name, graphUsed in CareAbout:
            # if not minimal_gifs:
                # gif_save_path = self.save_social_influence_diagram(
                    # graph=graphUsed,
                    # aliveMask=aliveMask,
                    # positionOfAgents=positionOfAgents,
                    # episode=total_num_steps,
                    # evaluation=f"Single{name}",
                # )
                # if gif_save_path is not None:
                    # data_dic[f"single_ep_eval/social_influence{name}_graph"] = (
                        # wandb.Image(gif_save_path)
                    # )

            PruneList = [pruning_amount_for_group_similarity]
            ###### Pruned
            if graph_gifs:
                for percentToPrune in PruneList:
                    secondary_pruned_gif_save_path = (
                        self.save_social_influence_diagram_pruning(
                            graph=graphUsed,
                            aliveMask=aliveMask,
                            positionOfAgents=positionOfAgents,
                            episode=total_num_steps,
                            evaluation=f"Pruning({percentToPrune}){name}",
                            PercentilePrune=percentToPrune,
                            AddRenderGraph=frames,
                            BigInfoOfSIPolicies=BigInfoOfSIPolicies,
                        )
                    )
                    if secondary_pruned_gif_save_path is not None:
                        data_dic[
                            f"single_ep_eval/social_influence{name}_graph_pruned({percentToPrune})"
                        ] = wandb.Image(secondary_pruned_gif_save_path)
                    if name != "_variance":
                        try:
                            (
                                graph_type_analysis_gif_save_path,
                                wandb_histogram,
                                HeatMapPath,
                            ) = self.graph_timestep_analysis(
                                graphUsed,
                                aliveMask,
                                episode=total_num_steps,
                                PercentilePrune=percentToPrune,
                                action_analysis=action_analysis,
                            )
                            if graph_type_analysis_gif_save_path is not None:
                                data_dic[
                                    f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})"
                                ] = wandb.Image(graph_type_analysis_gif_save_path)
                            data_dic[
                                f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})_histogram"
                            ] = wandb_histogram
                            if HeatMapPath is not None:
                                data_dic[
                                    f"single_ep_eval/GraphTypeJointAction_Heatmap_SI{name}_pruned({percentToPrune})"
                                ] = wandb.Image(HeatMapPath)
                        except Exception as e:
                            print("Graph Type Analysis Error", e)

            if not minimal_gifs:
                for window in range(
                    1, 4
                ):  # This is 2 to 5 because minimum 1 for past timestep
                    reciprocity_gif_save_path = None
                    learning_reciprocity = self.SI_reciprocity_analysis(
                        graphUsed, aliveMask, horizon=window
                    )
                    # reciprocity_gif_save_path = self.save_reciprocity_rollout(
                    #     learning_reciprocity,
                    #     aliveMask,
                    #     None,
                    #     total_num_steps,
                    #     horizon=window,
                    #     typeName=name,
                    # )

                    # if reciprocity_gif_save_path is not None:
                    #     data_dic[
                    #         f"single_ep_eval/reciprocity{name}_graph_(Horizon_{window+1})"
                    #     ] = wandb.Image(reciprocity_gif_save_path)

                    data_dic[
                        f"single_ep_eval/reciprocity{name}_across_timestep (Horizon_{window+1})"
                    ] = np.mean(learning_reciprocity)
                    print("HERE")
        if not minimal_gifs:
            # Not really that useful to have coordination bond info
            coordination_metrics = self.graph_coordination_analysis(
                graph, aliveMask, total_num_steps, PercentilePrune=percentToPrune
            )

            data_dic.update(coordination_metrics)

        meanSI_equalityRwd = np.mean(perAgentSocialInfluenceRewards)
        # self.buffer.set_group_social_influence_rewards(perAgentSocialInfluenceRewards)
        metrics = {
            "single_ep_eval/social_influence_metric_equality": meanSI_equalityRwd,
            "single_ep_eval/social_influence_variance_mean": np.mean(variance_graph),
            "single_ep_eval/social_influence_metric_mean": np.mean(graph),
        }

        metrics.update(data_dic)

        self.CentralMemory["internalCounter"] += 1
        NumberOfTimeCentralMemoryFilled = self.CentralMemory.get("internalCounter")
        if NumberOfTimeCentralMemoryFilled % 50 == 0:
            SummaryHeatMap, WHist, big_plot_of_graph_types = self.plot_central_memory(
                episode=total_num_steps, percentileToPruneList=PruneList
            )
            metrics[
                f"single_ep_eval/Summary-Graph_Types_social_influence_metric_pruned({percentToPrune})_histogram"
            ] = WHist
            metrics[
                f"single_ep_eval/Summary-GraphTypeJointAction_Heatmap_SI_metric_pruned({percentToPrune})"
            ] = wandb.Image(SummaryHeatMap)
            metrics[
                f"single_ep_eval/GraphType_Encountered_pruned({percentToPrune})"
            ] = wandb.Image(big_plot_of_graph_types)

        return metrics  # , gif_save_path, learning_gif_save_path_rep

    def get_graph_types(self, graph, aliveMask, PercentilePrune=35):
        """Get the graph types"""

        ### Clean up the graph
        flattenGraph = graph.flatten()
        flattenGraphCleaned = flattenGraph[flattenGraph > 1e-8]
        assert (
            sum(flattenGraphCleaned) > 0
        ), "There should be some social influence in the graph"
        TwentyFifthPercentile = np.percentile(
            flattenGraphCleaned, PercentilePrune
        )  # This is per episode
        MinSIForAllTimeSteps = TwentyFifthPercentile

        graph = np.where(graph < MinSIForAllTimeSteps, 0, graph)

        ### Unfold
        ListByTimestep = unstack(
            graph, 2
        )  # 5, 5, 3200 where 5 is the number of agents, 3200 is the number of timesteps

        graph_types = []
        for count, timestep_graph in enumerate(ListByTimestep):
            aliveMaskTimestep = aliveMask[count]
            aliveMaskTimestep_flat = aliveMaskTimestep.flatten()
            numberLivingAgents = int(np.sum(aliveMaskTimestep))
            assert (
                numberLivingAgents <= self.num_agents
            ), "All agents should be less than the total number of agents"
            if numberLivingAgents >= 2:
                G = nx.DiGraph()
                G.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes
                living_agents = [
                    i for i in range(self.num_agents) if aliveMaskTimestep_flat[i] == 1
                ]
                for cx, i in enumerate(living_agents):
                    # add number to agent nodes
                    G.nodes[cx]["label"] = i
                    for cy, j in enumerate(living_agents):
                        weight = timestep_graph[i][j]
                        # Prune the weights below a certain threshold (TwentyFifthPercentile)
                        if (
                            i != j
                            and weight != 0
                            and weight > 1e-8
                            and weight > TwentyFifthPercentile
                        ):
                            G.add_weighted_edges_from([(cx, cy, weight)])
                GraphType = self.si_graph_checker.check_which_graph(
                    G
                )  # returns a tuple
                graph_types.append(GraphType)

        return graph_types

    def plotTimestepGraphRewards(
        self, rewards, action_analysis=None, episode=None, extra_string=""
    ):
        def prepare_action_strings(actions):
            return [", ".join(info) if info else "No Actions" for info in actions]

        SavePath = None
        if action_analysis is not None:
            actionsTaken = [action_analysis[i] for i in range(len(action_analysis))]
            PreppedStrings = prepare_action_strings(actionsTaken)

        rewards = unstack(rewards, 0)  # makes the np array a list of values

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(rewards, marker="")

            if action_analysis is not None:
                for x, (y, ActionList) in enumerate(zip(rewards, actionsTaken)):
                    InfoString = PreppedStrings[x]
                    # ax.plot(x, y, marker=getMarkers(ActionList), label=InfoString, zorder=3, markersize=20)
                    # TODO
                    # See if we need the line above
                    ax.annotate(
                        InfoString,
                        (x, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        rotation=55,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                    )

                lines_labels = {
                    line.get_label(): line
                    for line in ax.get_lines()
                    if not line.get_label().startswith("_line")
                }
                ax.legend(
                    lines_labels.values(),
                    lines_labels.keys(),
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    labelspacing=2,
                )
            else:
                ax.plot(rewards, marker="o")

            ax.set_xlabel("Timesteps")
            ax.set_ylabel(f"Graph Similarity Rewards {extra_string}")
            ax.set_yticks(np.unique(rewards))
            ax.set_xticks(np.arange(len(rewards)))

            # Add margins to see all annotations
            ax.margins(x=0.1, y=0.1)

            ax.set_title(
                f"Social Influence Graph Equality {extra_string} Rewards over Ep (T{episode})"
            )

            SavePath = f"{self.gifs_dir}/SI_GraphType-Ep{episode}-{extra_string}.png"
            plt.tight_layout()
            plt.savefig(SavePath, bbox_inches="tight")
            plt.close(fig)

        return SavePath

    def graph_coordination_analysis(
        self, graph, aliveMask, total_num_steps, PercentilePrune=35
    ):
        """Analyse the coordination of the graph"""
        # Get the graph types
        graph_types = self.get_graph_types(graph, aliveMask, PercentilePrune)
        # print("Graph Types", graph_types)
        # Get the coordination metrics
        ## ScoreInfo [12, 9, 6, 2, 2, 1, 2, 2, 2, 6, 6, 8, 9, 9, 9, 12, 12, 9]
        ScoreInfo = [
            self.si_graph_checker.get_full_score(graph_type)
            for graph_type in graph_types
        ]
        # print("ScoreInfo", ScoreInfo)
        score_by_dual_edge, bondCounts = zip(*ScoreInfo)

        # TotalTimesteps = graph.shape[2]
        MoreThanTwoAgentsAliveTimesteps = sum(
            [1 for aliveMaskTimestep in aliveMask if int(np.sum(aliveMaskTimestep)) > 1]
        )

        singleBondCounts, doubleBondCounts = zip(*bondCounts)

        EpSingleBondCounts = np.sum(singleBondCounts)
        EpDoubleBondCounts = np.sum(doubleBondCounts)

        # TODO Count the number of times each graph type appears - but then consistency with actions?

        coordination_metrics = {
            "single_ep_eval/GraphType_CoordinationScore_FactorScore(Ep_Mean)": np.sum(
                score_by_dual_edge
            )
            / MoreThanTwoAgentsAliveTimesteps,
            "single_ep_eval/GraphType_CoordinationScore_BondCounts(Ep_Mean)": (
                EpSingleBondCounts + EpDoubleBondCounts
            )
            / MoreThanTwoAgentsAliveTimesteps,
            "single_ep_eval/GraphType_DoubleBondCounts(Ep_Mean)": EpDoubleBondCounts
            / MoreThanTwoAgentsAliveTimesteps,
            "single_ep_eval/GraphType_SingleBondCounts(Ep_Mean)": EpSingleBondCounts
            / MoreThanTwoAgentsAliveTimesteps,
            "single_ep_eval/GraphType_SingleDoubleBondRatio": EpSingleBondCounts
            / EpDoubleBondCounts,
        }

        return coordination_metrics

    def compute_dual_SI_group_social_influence_diagram_evaluation_single_episode(
        self,
        obs,
        actions,
        TeamBObs,
        positionOfAgents=None,
        frames=None,
        policiesA=None,
        policiesB=None,
        PlottingPolicies=False,
        total_num_steps=0,
    ):
        # note actions are a shared common
        assert self.run_mixed_population, "This function is only for mixed population"

        self.social_influence_net_A.calc_social_influence_reward_group(obs, actions)
        self.social_influence_net_B.calc_social_influence_reward_group(
            TeamBObs, actions
        )

        A_metrics = self.social_influence_net_A.calc_social_influence_reward_group(
            obs, actions
        )
        A_perAgentSocialInfluenceRewards, A_Graph = A_metrics[0]

        B_metrics = self.social_influence_net_B.calc_social_influence_reward_group(
            TeamBObs, actions
        )
        B_perAgentSocialInfluenceRewards, B_Graph = B_metrics[0]

        ##################### shared
        reformat_actions = convert_individual_to_joint(actions)
        aliveMask = np.ones_like(reformat_actions)
        # if action = 0 then agent is dead
        # aliveMask[reformat_actions == 0] = 0 #TODO All agents are alive throughout the game

        action_analysis = self.smac_action_analysis(actions)

        if policiesA is None or not PlottingPolicies:
            BigInfoOfSIPoliciesA = None
        else:
            CounterFactualInfoA = self.social_influence_net_A.calc_social_influence_reward_with_counterfactuals_returned(
                obs, actions
            )
            BigInfoOfSIPoliciesA = (CounterFactualInfoA, actions, policies)

        if policiesB is None or not PlottingPolicies:
            BigInfoOfSIPoliciesB = None
        else:
            CounterFactualInfoB = self.social_influence_net_B.calc_social_influence_reward_with_counterfactuals_returned(
                TeamBObs, TeamBActions
            )
            BigInfoOfSIPoliciesB = (CounterFactualInfoB, actions, policiesB)

        metrics = {}

        PruneList = [35]
        for percentToPrune in PruneList:
            pruned_gif_save_path = self.save_twin_social_influence_diagram_pruning(
                graph=[A_Graph, B_Graph],
                aliveMask=aliveMask,
                positionOfAgents=positionOfAgents,
                episode=total_num_steps,
                evaluation=f"TwinPruned({percentToPrune})_metric",
                PercentilePrune=percentToPrune,
                AddRenderGraph=frames,
                BigInfoOfSIPolicies=[BigInfoOfSIPoliciesA, BigInfoOfSIPoliciesB],
            )
            if pruned_gif_save_path is not None:
                metrics[
                    f"single_ep_eval/Two_Teams_social_influence_metric_graph_pruned({percentToPrune})"
                ] = wandb.Image(pruned_gif_save_path)

        return metrics

    def compute_dual_SI_group_social_influence_rewards_evaluation_single_episode(
        self,
        obs,
        actions,
        total_num_steps=0,
        positionOfAgents=None,
        minimal_gifs=False,
        frames=None,
    ):

        learning_metrics = self.social_influence_net.calc_social_influence_reward_group(
            obs, actions
        )
        learning_perAgentSocialInfluenceRewards, learning_Graph = learning_metrics[0]
        _, learning_variance_Graph = learning_metrics[1]

        fixed_metrics = (
            self.social_influence_net_fixed.calc_social_influence_reward_group(
                obs, actions
            )
        )
        fixed_perAgentSocialInfluenceRewards, fixed_Graph = fixed_metrics[0]
        _, fixed_variance_Graph = fixed_metrics[1]

        reformat_actions = convert_individual_to_joint(actions)
        aliveMask = np.ones_like(reformat_actions)
        # if action = 0 then agent is dead
        # aliveMask[reformat_actions == 0] = 0 #TODO All agents are alive through the game

        action_analysis = self.smac_action_analysis(actions)

        data_dic = {}

        learning_analysis = self.SI_analysis(learning_graph, aliveMask, total_num_steps)
        fixed_analysis = self.SI_analysis(fixed_graph, aliveMask, total_num_steps)

        # append (learning) or (fixed) to every key
        learning_analysis = {
            f"{key}(learning)": value for key, value in learning_analysis.items()
        }
        fixed_analysis = {
            f"{key}(fixed)": value for key, value in fixed_analysis.items()
        }

        data_dic.update(learning_analysis)
        data_dic.update(fixed_analysis)

        for name, learning_graph, fixed_graph in [
            ("_metric", learning_Graph, fixed_Graph),
            ("_variance", learning_variance_Graph, fixed_variance_Graph),
        ]:
            # last two
            IgnoreIf = minimal_gifs and name == "_variance"
            if not IgnoreIf:
                learning_gif_save_path_si = self.save_social_influence_diagram(
                    graph=learning_graph,
                    aliveMask=aliveMask,
                    positionOfAgents=positionOfAgents,
                    episode=total_num_steps,
                    evaluation=f"DualSIComparison-Learning{name}",
                )

                fixed_gif_save_path_si = self.save_social_influence_diagram(
                    graph=fixed_graph,
                    aliveMask=aliveMask,
                    positionOfAgents=positionOfAgents,
                    episode=total_num_steps,
                    evaluation=f"DualSIComparison-Fixed{name}",
                )

                if learning_gif_save_path_si is not None:
                    data_dic[
                        f"single_ep_eval/learning_model{name}_social_influence_graph"
                    ] = wandb.Image(learning_gif_save_path_si)

                if fixed_gif_save_path_si is not None:
                    data_dic[
                        f"single_ep_eval/fixed_model{name}_social_influence_graph"
                    ] = wandb.Image(fixed_gif_save_path_si)

            # if minimal_gifs:
            #     PruneList = [35]
            # else:
            #     PruneList = [35, 60]

            PruneList = [35]
            ###### Pruned
            for percentToPrune in PruneList:
                learning_gif_save_path_si_pruned = (
                    self.save_social_influence_diagram_pruning(
                        graph=learning_graph,
                        aliveMask=aliveMask,
                        positionOfAgents=positionOfAgents,
                        episode=total_num_steps,
                        evaluation=f"Pruning({percentToPrune})-Learning{name}",
                        PercentilePrune=percentToPrune,
                        AddRenderGraph=frames,
                    )
                )

                fixed_gif_save_path_si_pruned = (
                    self.save_social_influence_diagram_pruning(
                        graph=fixed_graph,
                        aliveMask=aliveMask,
                        positionOfAgents=positionOfAgents,
                        episode=total_num_steps,
                        evaluation=f"Pruning({percentToPrune})-Fixed{name}",
                        PercentilePrune=percentToPrune,
                        AddRenderGraph=frames,
                    )
                )

                if learning_gif_save_path_si_pruned is not None:
                    data_dic[
                        f"single_ep_eval/learning_model_social_influence{name}_graph_pruned({percentToPrune})"
                    ] = wandb.Image(learning_gif_save_path_si_pruned)

                if fixed_gif_save_path_si_pruned is not None:
                    data_dic[
                        f"single_ep_eval/fixed_model_social_influence{name}_graph_pruned({percentToPrune})"
                    ] = wandb.Image(fixed_gif_save_path_si_pruned)

                if name != "_variance":
                    graph_type_analysis_gif_save_path, wandb_histogram, HeatMapPath = (
                        self.graph_timestep_analysis(
                            learning_graph,
                            aliveMask,
                            episode=total_num_steps,
                            PercentilePrune=percentToPrune,
                            action_analysis=action_analysis,
                        )
                    )
                    if graph_type_analysis_gif_save_path is not None:
                        data_dic[
                            f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})(learning)"
                        ] = wandb.Image(graph_type_analysis_gif_save_path)
                    data_dic[
                        f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})_histogram(learning)"
                    ] = wandb_histogram
                    data_dic[
                        f"single_ep_eval/GraphTypeJointAction_Heatmap_SI{name}_pruned({percentToPrune})(learning)"
                    ] = wandb.Image(HeatMapPath)

                    graph_type_analysis_gif_save_path, wandb_histogram, HeatMapPath = (
                        self.graph_timestep_analysis(
                            fixed_graph,
                            aliveMask,
                            episode=total_num_steps,
                            PercentilePrune=percentToPrune,
                            action_analysis=action_analysis,
                        )
                    )
                    if graph_type_analysis_gif_save_path is not None:
                        data_dic[
                            f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})(fixed)"
                        ] = wandb.Image(graph_type_analysis_gif_save_path)
                    data_dic[
                        f"single_ep_eval/Graph_Types_social_influence{name}_pruned({percentToPrune})_histogram(fixed)"
                    ] = wandb_histogram
                    data_dic[
                        f"single_ep_eval/GraphTypeJointAction_Heatmap_SI{name}_pruned({percentToPrune})(fixed)"
                    ] = wandb.Image(HeatMapPath)

            if not minimal_gifs:
                for window in range(
                    1, 4
                ):  # This is 2 to 5 because minimum 1 for past timestep
                    learning_reciprocity = self.SI_reciprocity_analysis(
                        learning_graph, aliveMask, horizon=window
                    )
                    fixed_reciprocity = self.SI_reciprocity_analysis(
                        fixed_graph, aliveMask, horizon=window
                    )

                    learning_gif_save_path_rep = self.save_reciprocity_rollout(
                        learning_reciprocity,
                        aliveMask,
                        positionOfAgents,
                        total_num_steps,
                        horizon=window,
                        typeName=name,
                    )
                    fixed_gif_save_path_rep = self.save_reciprocity_rollout(
                        fixed_reciprocity,
                        aliveMask,
                        positionOfAgents,
                        total_num_steps,
                        horizon=window,
                        typeName=name,
                    )

                    if learning_gif_save_path_rep is not None:
                        data_dic[
                            f"single_ep_eval/learning_model_reciprocity{name}_graph_(Horizon_{window+1})"
                        ] = wandb.Image(learning_gif_save_path_rep)

                    if fixed_gif_save_path_rep is not None:
                        data_dic[
                            f"single_ep_eval/fixed_model_reciprocity{name}_graph_(Horizon_{window+1})"
                        ] = wandb.Image(fixed_gif_save_path_rep)

                    data_dic[
                        f"single_ep_eval/learning_model_reciprocity{name}_across_timestep (Horizon_{window+1})"
                    ] = np.mean(learning_reciprocity)
                    data_dic[
                        f"single_ep_eval/fixed_model_reciprocity{name}_across_timestep (Horizon_{window+1})"
                    ] = np.mean(fixed_reciprocity)

            data_dic[f"single_ep_eval/learning_model_social_influence_mean{name}"] = (
                np.mean(learning_graph)
            )
            data_dic[f"single_ep_eval/fixed_model_social_influence_mean{name}"] = (
                np.mean(fixed_graph)
            )
        # print("Mean Rewards")

        if learning_perAgentSocialInfluenceRewards.size > 0:
            learning_meanSIRwd = np.mean(learning_perAgentSocialInfluenceRewards)
        else:
            learning_meanSIRwd = 0

        if fixed_perAgentSocialInfluenceRewards.size > 0:
            fixed_meanSIRwd = np.mean(fixed_perAgentSocialInfluenceRewards)
        else:
            fixed_meanSIRwd = 0

        metrics = {
            "single_ep_eval/learning_model_social_influence_equality_metric": learning_meanSIRwd,
            "single_ep_eval/fixed_model_social_influence_equality_metric": fixed_meanSIRwd,
        }

        metrics.update(data_dic)

        return metrics

    def SI_reciprocity_analysis(self, graph, aliveMask, horizon=1):
        """See the reciprocity between timesteps

        horizon = 1 means that we are comparing the current timestep with the previous timestep
        """
        # Split the graphs into rollout timesteps
        ListByTimestep = unstack(
            graph, 2
        )  # 5, 5, 3200 where 5 is the number of agents, 3200 is the number of timesteps

        length = len(ListByTimestep)
        if length > self.episode_length:
            num_chunks = math.ceil(length / self.episode_length)
            split_list = [
                ListByTimestep[i * self.episode_length : (i + 1) * self.episode_length]
                for i in range(num_chunks)
            ]

        else:
            split_list = [ListByTimestep]

        ReciprocityRollout = []
        totalcount = 0
        for rollout in split_list:
            # print("Rollout", len(rollout))
            ReciprocityList = []
            # past_graph = None
            past_graphs = []
            for count, timestep_graph in enumerate(rollout):
                aliveMaskTimestep = aliveMask[totalcount, :, :]
                totalcount += 1
                # print(timestep_graph.shape) # (3, 3)
                numberLivingAgents = int(np.sum(aliveMaskTimestep))
                # print("numberLivingAgents", numberLivingAgents)
                if numberLivingAgents < 2:
                    # print("Less than 2 agents alive at timestep", count)
                    # quit if there are less than 2 agents
                    break
                assert (
                    len(past_graphs) <= horizon
                ), "Past graphs should always be less than or equal to horizon"
                if count != 0 and len(past_graphs) == horizon:
                    # Do the check against past graph
                    reciprocity = np.zeros((self.num_agents, self.num_agents))
                    for i in range(self.num_agents):
                        for j in range(self.num_agents):
                            if i != j:
                                # reciprocity[i][j] = timestep_graph[i][j] - past_graph[j][i]
                                past_graph = past_graphs[0]
                                reciprocity[i][j] = geo_mean_overflow(
                                    [timestep_graph[i][j], past_graph[j][i]]
                                )
                    ReciprocityList.append(reciprocity)
                    past_graphs.pop(0)

                past_graphs.append(timestep_graph)
            ReciprocityRollout.append(ReciprocityList)

        return ReciprocityRollout

    def SI_analysis(self, graph, aliveMask, episode=0):
        """Expects graph to be a 3D array of shape (num_agents, num_agents, num_timesteps). Going to assume that this is just one episode to be evaluated."""
        metrics = {}

        # if agents is dead, set SI from it and to it to zero, where graph to be a 3D array of shape (num_agents, num_agents, num_timesteps) and aliveMask to be a 2D array of shape (num_timesteps, num_agents, 1)
        # print("TODO aliveMask shape", aliveMask.shape) # (24, 3, 1)

        ############# for each timestep get the max social influence --- Max Pooling
        maxSI = np.max(graph, axis=(0, 1))
        meanMaxSI = np.mean(maxSI)

        metrics["SI Metrics/average_max_social_influence_per_timestep"] = meanMaxSI

        ############# threshold for episode (quartile)
        flattenGraph = graph.flatten()
        if False:  # whether to save the percentiles
            TwentyFifthPercentile = np.percentile(flattenGraph, 25)
            FiftyPercentile = np.percentile(flattenGraph, 50)
            SeventyFifthPercentile = np.percentile(flattenGraph, 75)

            metrics["SI Metrics/25th_percentile_social_influence"] = (
                TwentyFifthPercentile
            )
            metrics["SI Metrics/50th_percentile_social_influence"] = FiftyPercentile
            metrics["SI Metrics/75th_percentile_social_influence"] = (
                SeventyFifthPercentile
            )

        ###### Plot this as wandb scatter where x is timestep and y is the agent SI
        social_influence_scores = np.mean(graph, axis=1)  # num_agents, num_timesteps
        # x = num_timesteps, y = num_agents
        # split by agent
        split_social_influence_scores = unstack(social_influence_scores, 0)

        plt.figure()
        # For each agent, plot a line
        for i in range(self.num_agents):
            plt.plot(split_social_influence_scores[i], label=f"Agent {i+1}")

        # Add a legend
        plt.legend()

        # Add labels for the x and y axes
        plt.xlabel("Timesteps")
        plt.ylabel("Influencing Agent Social Influence Enacted")
        fileName = f"{self.gifs_dir}/Analysis-PerAgentInfluence-TS{episode}.png"

        num_timesteps = len(split_social_influence_scores[0])
        plt.xticks(range(num_timesteps))  # Set tick marks for every timestep

        plt.title(f"Social Influence Enacted for Timestep {episode}")

        plt.savefig(fileName)
        metrics.update(
            {"SI Metrics/per_agent_influence_enacted": wandb.Image(fileName)}
        )

        # clear the graph
        plt.clf()

        # show distribution of values histogram si influence

        # remove 0 and 1e-8 values
        cleanedGraph = flattenGraph[flattenGraph > 1e-8]

        plt.hist(cleanedGraph, bins="auto", density=True)
        plt.title(f"Social Influence Distribution for Timestep {episode}")
        plt.xlabel("Social Influence")
        plt.ylabel("Frequency")
        plt.savefig(
            f"{self.gifs_dir}/Analysis-SocialInfluenceDistribution-TS{episode}.png"
        )
        metrics.update(
            {
                "SI Metrics/social_influence_distribution": wandb.Image(
                    f"{self.gifs_dir}/Analysis-SocialInfluenceDistribution-TS{episode}.png"
                )
            }
        )

        # clear the graph
        plt.clf()

        # metrics.update({
        #     "SI Metrics/per_agent_influence": wandb.plot.line_series(
        #         xs= np.arange(social_influence_scores.shape[1]), # All the timesteps
        #         ys=split_social_influence_scores,
        #         keys=["Agent " + str(i) for i in range(self.num_agents)],
        #         title="Social Influence per Agent")
        # })

        social_attention_scores = np.mean(graph, axis=0)  # num_agents, num_timesteps
        # x = num_timesteps, y = num_agents
        # split by agent
        split_social_attention_scores = unstack(social_attention_scores, 0)

        plt.figure()
        # For each agent, plot a line
        for i in range(self.num_agents):
            plt.plot(split_social_attention_scores[i], label=f"Agent {i+1}")

        # Add a legend
        plt.legend()

        # Add labels for the x and y axes
        plt.xlabel("Timesteps")
        plt.ylabel("Receiving Agent Social Influence")

        num_timesteps = len(split_social_attention_scores[0])
        plt.xticks(range(num_timesteps))  # Set tick marks for every timestep

        fileName = f"{self.gifs_dir}/Analysis-PerAgentAttentionReceived-TS{episode}.png"

        plt.title(f"Social Influence Received for Timestep {episode}")

        plt.savefig(fileName)
        metrics.update(
            {"SI Metrics/per_agent_influence_received": wandb.Image(fileName)}
        )

        # clear the graph
        plt.clf()

        # metrics.update({
        #     "SI Metrics/per_agent_attention": wandb.plot.line_series(
        #         xs= np.arange(social_attention_scores.shape[1]),  # All the timesteps
        #         ys=split_social_attention_scores,
        #         keys=["Agent " + str(i) for i in range(self.num_agents)],
        #         title="Social Attention per Agent")
        # })

        plt.close("all")
        plt.cla()

        return metrics

    def save_reciprocity_rollout(
        self,
        ReciprocityRollout,
        aliveMask,
        positionOfAgents=None,
        episode=0,
        horizon=1,
        typeName="",
    ):
        """Given a graph of X-1 timesteps, with a graph of agent to agent reciprocity between timesteps, make the agents nodes, and show the weights as edges, then save the graph"""
        CurrentRollOut = []
        rollout = 0
        count = 0

        GraphProduced = False

        numberOfOtherAgents = self.num_agents - 1

        split_aliveList = unstack(aliveMask, 2)
        length = len(split_aliveList)
        num_chunks = math.ceil(length / self.episode_length)

        ReciprocityRolloutArray = np.array(ReciprocityRollout)
        actualMaxSI = np.max(ReciprocityRolloutArray)
        MaxSIForAllTimeSteps = max(
            actualMaxSI, 0.01 * numberOfOtherAgents
        )  # 1 #np.max(graph)
        MinSIForAllTimeSteps = 0

        # Remove the first alive mask since we are not comparing the first timestep to anything
        gif_save_path = None
        totalcount = 0
        for rollout in ReciprocityRollout:
            for count, timestep_graph in enumerate(rollout):
                aliveMaskTimestep = aliveMask[totalcount, :, :]
                totalcount += 1
                numberLivingAgents = int(np.sum(aliveMaskTimestep))

                if numberLivingAgents < 2:
                    # quit if there are less than 2 agents
                    break

                aliveMaskTimestep_flat = aliveMaskTimestep.flatten()

                AliveAgentnode = np.where(aliveMaskTimestep == 1)[0]
                G = nx.DiGraph()
                G.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes

                # node_labels_alive is a list of all the indicies of alive agents according to where they are in aliveMaskTimestep (where aliveMaskTimestep == 1)
                # node_labels_alive = [i for i in range(timestep_graph.shape[0]) if aliveMaskTimestep_flat[i] == 1]
                living_agents = [
                    i for i in range(self.num_agents) if aliveMaskTimestep_flat[i] == 1
                ]
                # print("Living agents", living_agents)
                for cx, i in enumerate(living_agents):
                    # add number to agent nodes
                    G.nodes[cx]["label"] = i
                    for cy, j in enumerate(living_agents):
                        weight = timestep_graph[i][j]
                        # Prune the weights below a certain threshold (TwentyFifthPercentile)
                        if i != j and weight != 0 and weight > 1e-8:
                            NewW = round(
                                weight, 3
                            )  # For better visualization round the weight
                            if NewW == 0:
                                NewW = round(weight, 7)
                            if NewW != 0:
                                G.add_weighted_edges_from([(cx, cy, NewW)])

                # Save fixed positins of nodes
                node_positions = nx.circular_layout(
                    G
                )  # Use this in case missing an agent
                if positionOfAgents is not None:
                    positionOfAgentsThisTime = positionOfAgents[
                        count + horizon, :, :
                    ]  # Fixed position to be of most recent timestep
                    # only get living agents positions - delete the dead agents
                    filteredPositionOfAgentsThisTime = positionOfAgentsThisTime[
                        AliveAgentnode
                    ]  # (numberLivingAgents, 2)
                    # make into format of networkx
                    node_positions.update(
                        {
                            i: (
                                filteredPositionOfAgentsThisTime[i][0],
                                filteredPositionOfAgentsThisTime[i][1],
                            )
                            for i in range(numberLivingAgents)
                        }
                    )

                else:
                    if self.graph_node_positions is None:
                        self.graph_node_positions = node_positions
                    else:
                        node_positions = self.graph_node_positions

                ## Do some adjusting of positions for better visibility
                node_positions = nx.spring_layout(G, pos=node_positions)

                #### Color of edges
                # Get the weights of the edges
                weights = nx.get_edge_attributes(G, "weight").values()

                # Normalize the weights to the range [0, 1]
                weights = np.array(list(weights))
                # weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())
                weights_normalized = (weights - MinSIForAllTimeSteps) / (
                    MaxSIForAllTimeSteps - MinSIForAllTimeSteps
                )
                # print(weights_normalized)
                # width = [10.0 * i for i in list(weights_normalized)]

                # Get the colors from the colormap
                # colors = plt.cm.viridis(weights_normalized)
                colorsEdges = plt.cm.plasma(weights_normalized)

                #### Draw the graph
                # nx.draw(G, with_labels=True, font_weight='bold', pos = node_positions, node_size=500, font_size=10, font_color='black', edge_color=colorsEdges, width=1.0, edge_cmap=plt.cm.Blues, arrows=True, connectionstyle='arc3, rad = 0.15')
                # nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color='black', font_size=8, label_pos=0.3)

                nx.draw(
                    G,
                    with_labels=True,
                    font_weight="bold",
                    pos=node_positions,
                    node_size=1000,
                    arrowsize=50,
                    font_size=12,
                    font_color="black",
                    edge_color=colorsEdges,
                    width=5,
                    edge_cmap=plt.cm.Blues,
                    arrows=True,
                    connectionstyle="arc3, rad = 0.15",
                )
                nx.draw_networkx_edge_labels(
                    G,
                    pos=node_positions,
                    edge_labels=nx.get_edge_attributes(G, "weight"),
                    font_color="black",
                    font_size=12,
                    label_pos=0.3,
                )

                # Create a colorbar
                if len(weights) > 0:
                    minimum = min(list(weights))
                    maximum = max(list(weights))

                    # Create a new axes for the colorbar
                    divider = make_axes_locatable(plt.gca())
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)

                    sm = plt.cm.ScalarMappable(
                        cmap=plt.cm.plasma,
                        norm=colors.Normalize(vmin=minimum, vmax=maximum),
                    )
                    # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=colors.Normalize(vmin=min(list(weights)), vmax=max(list(weights))))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")

                    # Set the colorbar ticks
                    cbar.set_ticks(np.unique(weights))

                    # Set the colorbar tick labels
                    cbar.set_ticklabels(np.unique(weights))

                    # Adjust size of colorbar
                    cbar.ax.tick_params(labelsize=8)
                    cbar.set_label("Reciprocity", rotation=0, labelpad=10, fontsize=10)

                else:
                    minimum = 0
                    maximum = 0.01 * numberOfOtherAgents
                    # print("Reciprocity: No weights for this timestep", count, "graph is", timestep_graph.shape, "aliveMask is", aliveMaskTimestep, "graph values", timestep_graph)

                # Name
                rollout = count // self.episode_length
                ts = count % self.episode_length

                fileName = f"{self.gifs_dir}/P-Horizon_{horizon}-Ep{episode}_R{rollout}_TS{ts}-reciprocity_graph.png"

                plt.title(
                    f"Reciprocity between Timestep {ts} and {ts+horizon} (Horizon {horizon+1})"
                )

                plt.savefig(fileName)
                # print("Saving ", fileName)
                CurrentRollOut.append(fileName)

                # clear the graph
                plt.clf()

                # plt.close(fig)

            # if this is the last timestep for this rollout, make a gif
            if (count + 1) // self.episode_length != rollout:
                gif_save_path = self.make_gif_from_images(
                    episode, horizon, CurrentRollOut, "Reciprocity" + typeName
                )
                CurrentRollOut = []
                GraphProduced = True

        if not GraphProduced:
            if len(CurrentRollOut) == 0:
                print("No images to make gif from.")
            else:
                # Max episode length not used since we are not using the rollout, single episode evaluation
                gif_save_path = self.make_gif_from_images(
                    episode, horizon, CurrentRollOut, "Reciprocity" + typeName
                )

        return gif_save_path  # only the last one because we just need one

    def save_social_influence_diagram(
        self, graph, aliveMask, positionOfAgents=None, episode=0, evaluation="False"
    ):
        """Given a graph of X timesteps, with a graph of agent to agent influence, make the agents nodes, and show the weights as edges, then save the graph"""
        CurrentRollOut = []
        rollout = 0
        count = 0

        GraphProduced = False
        assert graph.shape[0] == self.num_agents, "shape check for graph"
        assert graph.shape[1] == self.num_agents, "shape check for graph"
        ListByTimestep = unstack(
            graph, 2
        )  # 5, 5, 3200 where 5 is the number of agents, 3200 is the number of timesteps
        numberOfOtherAgents = self.num_agents - 1
        actualMaxSI = np.max(ListByTimestep)
        MaxSIForAllTimeSteps = max(
            actualMaxSI, 0.01 * numberOfOtherAgents
        )  # 1 #np.max(graph)
        MinSIForAllTimeSteps = 0  # np.min(graph)
        # print("aliveMask", aliveMask.shape)
        # print("graph", graph.shape)
        gif_save_path = None
        for count, timestep_graph in enumerate(ListByTimestep):
            aliveMaskTimestep = aliveMask[count, :, :]
            aliveMaskTimestep_flat = aliveMaskTimestep.flatten()
            # print("aliveMaskTimestep", aliveMaskTimestep.shape)

            numberLivingAgents = int(np.sum(aliveMaskTimestep))
            assert (
                numberLivingAgents <= self.num_agents
            ), "All agents should be less than the total number of agents"

            if numberLivingAgents >= 2:

                # EmptyAgentNodes = timestep_graph.shape[0]
                # Assuming node_labels is your tensor containing node labels
                # and aliveMaskTimestep is a tensor of the same shape containing boolean values

                AliveAgentnode = np.where(aliveMaskTimestep == 1)[0]
                G = nx.DiGraph()
                G.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes

                living_agents = [
                    i for i in range(self.num_agents) if aliveMaskTimestep_flat[i] == 1
                ]
                # print("Living agents", living_agents)
                for cx, i in enumerate(living_agents):
                    # add number to agent nodes
                    G.nodes[cx]["label"] = i
                    for cy, j in enumerate(living_agents):
                        weight = timestep_graph[i][j]
                        # Prune the weights below a certain threshold (TwentyFifthPercentile)
                        if i != j and weight != 0 and weight > 1e-8:
                            NewW = round(
                                weight, 3
                            )  # For better visualization round the weight
                            if NewW == 0:
                                NewW = round(weight, 7)
                            if NewW != 0:
                                G.add_weighted_edges_from([(cx, cy, NewW)])

                # Save fixed positins of nodes
                node_positions = nx.circular_layout(G)
                if positionOfAgents is not None:
                    # node_positions = nx.circular_layout(G) # Use this in case missing an agent
                    positionOfAgentsThisTime = positionOfAgents[count, :, :]
                    # only get living agents positions - delete the dead agents
                    filteredPositionOfAgentsThisTime = positionOfAgentsThisTime[
                        AliveAgentnode
                    ]  # (numberLivingAgents, 2)
                    # print("filteredPositionOfAgentsThisTime", filteredPositionOfAgentsThisTime.shape)
                    # make into format of networkx
                    node_positions.update(
                        {
                            i: (
                                filteredPositionOfAgentsThisTime[i][0],
                                filteredPositionOfAgentsThisTime[i][1],
                            )
                            for i in range(numberLivingAgents)
                        }
                    )

                else:
                    if self.graph_node_positions is None:
                        # node_positions = nx.circular_layout(G)
                        self.graph_node_positions = node_positions
                    else:
                        node_positions = self.graph_node_positions

                ## Do some adjusting of positions for better visibility
                node_positions = nx.spring_layout(G, pos=node_positions)

                #### Color of edges
                # Get the weights of the edges
                weights = nx.get_edge_attributes(G, "weight").values()

                # Normalize the weights to the range [0, 1]
                weights = np.array(list(weights))

                # weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())
                weights_normalized = (weights - MinSIForAllTimeSteps) / (
                    MaxSIForAllTimeSteps - MinSIForAllTimeSteps
                )
                # print(weights_normalized)
                # width = [10.0 * i for i in list(weights_normalized)]

                # Get the colors from the colormap
                # colors = plt.cm.viridis(weights_normalized)
                colorsEdges = plt.cm.plasma(weights_normalized)

                #### Draw the graph
                # nx.draw(G, with_labels=True, font_weight='bold', pos = node_positions, node_size=500, font_size=10, font_color='black', edge_color=colorsEdges, width=1.0, edge_cmap=plt.cm.Blues, arrows=True, connectionstyle='arc3, rad = 0.15')
                # nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color='black', font_size=8, label_pos=0.3)

                nx.draw(
                    G,
                    with_labels=True,
                    font_weight="bold",
                    pos=node_positions,
                    node_size=1000,
                    arrowsize=50,
                    font_size=12,
                    font_color="black",
                    edge_color=colorsEdges,
                    width=5,
                    edge_cmap=plt.cm.Blues,
                    arrows=True,
                    connectionstyle="arc3, rad = 0.15",
                )
                nx.draw_networkx_edge_labels(
                    G,
                    pos=node_positions,
                    edge_labels=nx.get_edge_attributes(G, "weight"),
                    font_color="black",
                    font_size=12,
                    label_pos=0.3,
                )

                # Create a new axes for the colorbar
                divider = make_axes_locatable(plt.gca())
                cax = divider.append_axes("bottom", size="5%", pad=0.05)

                if len(weights) > 0:
                    minimum = min(list(weights))
                    maximum = max(list(weights))

                    sm = plt.cm.ScalarMappable(
                        cmap=plt.cm.plasma,
                        norm=colors.Normalize(vmin=minimum, vmax=maximum),
                    )

                    # Create a colorbar
                    # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=colors.Normalize(vmin=min(list(weights)), vmax=max(list(weights))))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")

                    # Set the colorbar ticks
                    cbar.set_ticks(np.unique(weights))

                    # Set the colorbar tick labels
                    cbar.set_ticklabels(np.unique(weights))

                    # Adjust size of colorbar
                    cbar.ax.tick_params(labelsize=8)
                    cbar.set_label(
                        "Social Influence", rotation=0, labelpad=10, fontsize=10
                    )
                # else:
                #     # minimum = 0
                #     # maximum = 0.01 * numberOfOtherAgents
                #     print("SI: No weights for this timestep", count, "graph is", timestep_graph.shape, "aliveMask is", aliveMaskTimestep.shape, "graph values", timestep_graph)

                # Name
                rollout = count // self.episode_length
                ts = count % self.episode_length

                fileName = f"{self.gifs_dir}/P-Ep{episode}_R{rollout}_TS{ts}-social_influence_graph.png"

                plt.title(f"Social Influence Timestep {ts}")

                plt.savefig(fileName)
                CurrentRollOut.append(fileName)

                # clear the graph
                plt.clf()
                # plt.close(fig)

            # if this is the last timestep for this rollout, make a gif
            if (count + 1) // self.episode_length != rollout:
                gif_save_path = self.make_gif_from_images(
                    episode, rollout, CurrentRollOut, evaluation
                )
                CurrentRollOut = []
                GraphProduced = True

        if not GraphProduced:
            if len(CurrentRollOut) == 0:
                print("No images to make gif from but this graph shape", graph.shape)
            else:
                # Max episode length not used since we are not using the rollout, single episode evaluation
                gif_save_path = self.make_gif_from_images(
                    episode, "-SingleEval", CurrentRollOut, evaluation
                )

        return gif_save_path  # only the last one because we just need one

    def make_gif_from_images(self, episode, rollout, imagesPaths, evaluation=False):
        """Takes in a list of image paths, and makes a gif from them with file name
        Ep {episode} R {rollout} social_influence_graph.gif"""

        # save the gif to a file
        if evaluation == "True":
            gif_save_path = (
                f"{self.gifs_dir}/Evaluation-TS{episode}_social_influence_graph"
            )
        elif "Single" in evaluation:
            gif_save_path = (
                f"{self.gifs_dir}/Eval-Render-TS{episode}_social_influence_graph"
            )
            if evaluation != "Single":
                gif_save_path = f"{self.gifs_dir}/Eval-Render-TS{episode}_SI{evaluation.replace('Single', '')}_graph"
        elif evaluation == "False":
            gif_save_path = (
                f"{self.gifs_dir}/Train-Ep{episode}_R{rollout}_social_influence_graph"
            )
        elif "DualSIComparison" in evaluation:
            gif_save_path = f"{self.gifs_dir}/{evaluation}/Eval-Render-TS{episode}_social_influence_graph"
        elif "Reciprocity" in evaluation:
            gif_save_path = f"{self.gifs_dir}/Eval-Render-TS{episode}_Horizon{rollout}_{evaluation}_graph"
        elif "Pruning" in evaluation:
            gif_save_path = f"{self.gifs_dir}/Eval-Render-TS{episode}_{evaluation}_social_influence_graph"
        elif "TwinPruned" in evaluation:
            gif_save_path = f"{self.gifs_dir}/Eval-Render-TS{episode}_{evaluation}_social_influence_graph"
        else:
            gif_save_path = (
                f"{self.gifs_dir}/Eval-Render-TS{episode}_social_influence_graph"
            )

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        gif_save_path += "_" + timestamp + ".gif"

        frames = [imageio.imread(f) for f in imagesPaths]
        if len(frames) == 0:
            return None
        imageio.mimwrite(uri=gif_save_path, ims=frames, fps=1)

        for f in imagesPaths:
            os.remove(f)

        if not os.path.exists(gif_save_path):
            return None

        return gif_save_path

    def select_social_influence_for_predicted_policies(
        self,
        CounterfactualInfo,
        timestep,
        ThinkerAgent,
        ThinkerAction,
        PredictedForAgent,
        TrueActionUsed,
        WhetherTrueAction,
    ):
        """BigInfoOfSIPolicies is a tuple of all_agent_social_influence, counterfactual_policies, pred_true_policies.
        all_agent_social_influence is a list of counterfactual JS Div Value for each agent, and counterfactual_policies is a list of counterfactual policies for each agent,
        and pred_true_policies is a list of predicted policies for true action for each agent.
        """

        si_values, counterfactual_policies, pred_true_policies = CounterfactualInfo

        # if ThinkerAction is a list of size 1, make int
        if not isinstance(ThinkerAction, int) and ThinkerAction.size == 1:
            ThinkerAction = ThinkerAction.item()
            # make float to int
            ThinkerAction = int(ThinkerAction)

        # ThinkerAgent is always giving the correct ID in other code (1, 3 etc)

        # Input are lists, which are by agent so index that first
        si_values = si_values[ThinkerAgent]
        counterfactual_policies = counterfactual_policies[ThinkerAgent]
        pred_true_policies = pred_true_policies[ThinkerAgent]

        # Now check the shapes --- this is already by thinking agent
        # 7 because no op and true action don't have a social influence value
        # print("pred_true_policies", pred_true_policies.shape) #pred_true_policies torch.Size([17, 2, 9]) timestep, other agents, policy size
        # print("si_values", si_values.shape) #si_values torch.Size([17, 7, 2]) # timestep, counterfactual actions, other agents
        # print("counterfactual_policies", counterfactual_policies.shape) #counterfactual_policies torch.Size([17, 7, 2, 9]) timestep, counterfactual actions, other agents, policy size

        # Make into numpy arrays
        si_values = si_values.numpy()
        counterfactual_policies = counterfactual_policies.numpy()
        pred_true_policies = pred_true_policies.numpy()

        ### Because it is giving the correct ID, I need to adjust here
        IndexOfPredictedAgent = PredictedForAgent
        if PredictedForAgent > ThinkerAgent:
            IndexOfPredictedAgent -= 1  # because we exclude the thinker agent from the list of agents in BigInfoOfSIPolicies

        ActionIndex = ThinkerAction - 1  # because we exclude the no op action

        if WhetherTrueAction:
            # if we are using the true action, (right hand side) we are using the predicted true policies
            # Sanity Check of whether agent is alive
            if pred_true_policies.shape[0] <= timestep:
                return None, None
            MyPolicy = pred_true_policies[timestep, IndexOfPredictedAgent, :]
            MySIValue = None

        else:
            if ThinkerAction >= TrueActionUsed:
                ActionIndex -= 1  # because we exclude the true action and have done a minus for zero index
            # Use the counterfactual policies
            # print("ThinkerAction", ThinkerAction, "ActionIndex", ActionIndex)
            # Sanity Check of whether agent is alive
            if counterfactual_policies.shape[0] <= timestep:
                return None, None
            MyPolicy = counterfactual_policies[
                timestep, ActionIndex, IndexOfPredictedAgent, :
            ]
            MySIValue = si_values[timestep, ActionIndex, IndexOfPredictedAgent]

        MyPolicy = MyPolicy.squeeze()
        # TODO handle case where agent is dead and there is no social influence not zero padded
        # print("DEBUGGING HERE")
        # print("MySIValue", MySIValue)
        # print("MyPolicy", MyPolicy)
        # print(f"Shape of si_values: {si_values.shape}, Shape of counterfactual_policies: {counterfactual_policies.shape}, Shape of pred_true_policies: {pred_true_policies.shape}")
        # Shape of si_values: (18, 7, 2), Shape of counterfactual_policies: (18, 7, 2, 9), Shape of pred_true_policies: (18, 2, 9)
        # assert isinstance(MySIValue, float) or MySIValue is None, f"MySI should be a float or None, it is {MySIValue}"
        assert isinstance(MyPolicy, np.ndarray), "MyPolicy should be a numpy array"
        # policy should be one dimension

        assert len(MyPolicy.shape) == 1, "MyPolicy should be one dimension"

        return MySIValue, MyPolicy

    def save_social_influence_diagram_pruning(
        self,
        graph,
        aliveMask,
        positionOfAgents=None,
        episode=0,
        evaluation="False",
        PercentilePrune=25,
        AddRenderGraph=None,
        BigInfoOfSIPolicies=None,
    ):
        """Given a graph of X timesteps, with a graph of agent to agent influence, make the agents nodes, and show the weights as edges, then save the graph"""
        CurrentRollOut = []
        rollout = 0
        count = 0

        AddCounterfactualPolicyAnnotations = BigInfoOfSIPolicies is not None
        if AddCounterfactualPolicyAnnotations and AddRenderGraph is not None:
            # Making the fullest version of the graph
            CounterFactualInfo, actions, policies = BigInfoOfSIPolicies
            # Makes sure both actions and policies are numpy arrays
            actions = np.array(actions)
            policies = np.array(policies)

        GraphProduced = False
        assert graph.shape[0] == self.num_agents, "shape check for graph"
        assert graph.shape[1] == self.num_agents, "shape check for graph"
        ListByTimestep = unstack(
            graph, 2
        )  # 5, 5, 3200 where 5 is the number of agents, 3200 is the number of timesteps
        numberOfOtherAgents = self.num_agents - 1
        actualMaxSI = np.max(ListByTimestep)
        MaxSIForAllTimeSteps = max(
            actualMaxSI, 0.01 * numberOfOtherAgents
        )  # 1 #np.max(graph)
        # MinSIForAllTimeSteps = 0 #np.min(graph)
        gif_save_path = None

        flattenGraph = graph.flatten()
        # Remove all the zeroes
        # flattenGraphCleaned = flattenGraph[flattenGraph != 0] # Should only keep the social influence values for the percentile
        flattenGraphCleaned = flattenGraph[flattenGraph > 1e-8]
        # flattenGraphCleaned = flattenGraph
        # flattenGraphCleaned[flattenGraphCleaned <= 1e-8] = 0

        assert (
            sum(flattenGraphCleaned) > 0
        ), "There should be some social influence in the graph"

        TwentyFifthPercentile = np.percentile(
            flattenGraphCleaned, PercentilePrune
        )  # This is per episode
        MinSIForAllTimeSteps = TwentyFifthPercentile

        NameThreshold = round(TwentyFifthPercentile, 4)

        ##### Clean up the weights for display
        DisplayGraph = np.where(graph < TwentyFifthPercentile, 0, graph)

        # Histogram equalization
        # DisplayGraph = np.where(DisplayGraph == 0, 0, np.log(DisplayGraph))
        ActivePoints = DisplayGraph != 0
        DisplayGraph = equalize_hist(DisplayGraph, nbins=256, mask=ActivePoints)

        DisplayListByTimestep = unstack(
            DisplayGraph, 2
        )  # 5, 5, 3200 where 5 is the number of agents, 3200 is the number of timesteps
        MaxSIForAllTimeSteps = max(np.max(DisplayListByTimestep), 1)  # 1 #np.max(graph)
        MinSIForAllTimeSteps = min(np.min(DisplayGraph), 0)  # should minimum zero

        # print("Number Of Timesteps", len(ListByTimestep))
        # print("Policy shape", policies.shape)
        # print("Actions shape", actions.shape)
        # si_values, counterfactual_policies, pred_true_policies = CounterFactualInfo
        # print("si_values shape", si_values[0].shape)
        # print("counterfactual_policies shape", counterfactual_policies[0].shape)
        # print("pred_true_policies shape", pred_true_policies[0].shape)
        # print("pred_true_policies shape", pred_true_policies[1].shape)
        # print("pred_true_policies shape", pred_true_policies[2].shape)

        NumberOfTimesteps = (
            len(ListByTimestep) - 2
        )  # We are plotting the next timestep for SI

        # print("Number of SI Timesteps", len(ListByTimestep)) # 22
        # print("Number of Frames", len(AddRenderGraph)) # 21

        for count in range(-2, NumberOfTimesteps):
            ShouldIPlotSI = count >= 0 and count + 1 < len(ListByTimestep)
            if ShouldIPlotSI:
                timestep_graph = ListByTimestep[count + 1]
                aliveMaskTimestep = aliveMask[count + 1, :, :]
                aliveMaskTimestep_flat = aliveMaskTimestep.flatten()
                display_timestep_graph = DisplayListByTimestep[count + 1]
                # print("aliveMaskTimestep", aliveMaskTimestep.shape)

                # CounterFactualInfoTS = None
                actionsTS = None
                policiesTS = None
                next_actionsTS = None
                next_policiesTS = None
                if AddCounterfactualPolicyAnnotations and AddRenderGraph is not None:
                    # print("CounterFactualInfo shape", CounterFactualInfo.shape)
                    # CounterFactualInfoTS = CounterFactualInfo[count]
                    # print("CounterFactualInfoTS shape", CounterFactualInfoTS.shape)
                    # print("actions shape", actions.shape) actions shape (3, 18, 1)
                    # print("policies shape", policies.shape) policies shape (3, 18, 9)
                    actionsTS = actions[:, count, :]
                    policiesTS = policies[:, count, :]
                    if count + 1 < len(ListByTimestep):
                        next_actionsTS = actions[:, count + 1, :]
                        next_policiesTS = policies[:, count + 1, :]

                numberLivingAgents = int(np.sum(aliveMaskTimestep))
                assert (
                    numberLivingAgents <= self.num_agents
                ), "All agents should be less than the total number of agents"

                if numberLivingAgents >= 2:
                    # Assuming node_labels is your tensor containing node labels
                    # and aliveMaskTimestep is a tensor of the same shape containing boolean values

                    AliveAgentnode = np.where(aliveMaskTimestep == 1)[0]
                    G = nx.DiGraph()
                    G.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes

                    living_agents = [
                        i
                        for i in range(self.num_agents)
                        if aliveMaskTimestep_flat[i] == 1
                    ]
                    # print("Living agents", living_agents)
                    for cx, i in enumerate(living_agents):
                        # add number to agent nodes
                        G.nodes[cx]["label"] = i
                        for cy, j in enumerate(living_agents):
                            weight = timestep_graph[i][j]
                            display_weight = display_timestep_graph[i][j]
                            # Prune the weights below a certain threshold (TwentyFifthPercentile)
                            if (
                                i != j
                                and weight != 0
                                and weight > 1e-8
                                and weight > TwentyFifthPercentile
                            ):
                                NewW = round(
                                    display_weight, 2
                                )  # For better visualization round the weight
                                if NewW == 0:
                                    NewW = round(display_weight, 3)
                                if NewW != 0:
                                    G.add_weighted_edges_from([(cx, cy, NewW)])

                    # Save fixed positins of nodes
                    node_positions = nx.circular_layout(G)
                    if positionOfAgents is not None:
                        # node_positions = nx.circular_layout(G) # Use this in case missing an agent
                        positionOfAgentsThisTime = positionOfAgents[count, :, :]
                        # only get living agents positions - delete the dead agents
                        filteredPositionOfAgentsThisTime = positionOfAgentsThisTime[
                            AliveAgentnode
                        ]  # (numberLivingAgents, 2)
                        # print("filteredPositionOfAgentsThisTime", filteredPositionOfAgentsThisTime.shape)
                        # make into format of networkx
                        node_positions.update(
                            {
                                i: (
                                    filteredPositionOfAgentsThisTime[i][0],
                                    filteredPositionOfAgentsThisTime[i][1],
                                )
                                for i in range(numberLivingAgents)
                            }
                        )

                    else:
                        if self.graph_node_positions is None:
                            # node_positions = nx.circular_layout(G)
                            self.graph_node_positions = node_positions
                        else:
                            node_positions = self.graph_node_positions

                    ## Do some adjusting of positions for better visibility
                    node_positions = nx.spring_layout(
                        G, pos=node_positions, iterations=6
                    )

                    #### Color of edges
                    # Get the weights of the edges
                    weights = nx.get_edge_attributes(G, "weight").values()

                    # Normalize the weights to the range [0, 1]
                    weights = np.array(list(weights))

                    # weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())
                    weights_normalized = (weights - MinSIForAllTimeSteps) / (
                        MaxSIForAllTimeSteps - MinSIForAllTimeSteps
                    )

                    colorsEdges = plt.cm.plasma(weights_normalized)

                #### Subplot to show graph type
                GraphType = self.si_graph_checker.check_which_graph(G)
            AddCounterfactualPolicyAnnotations = BigInfoOfSIPolicies is not None
            if (
                AddCounterfactualPolicyAnnotations
                and AddRenderGraph is not None
                and count + 2 < len(AddRenderGraph)
                and count >= 0
                and self.num_agents < 5
            ):
                # Making the fullest version of the graph
                # BigInfoOfSIPolicies = Agent prediction for other agents's policies, for each agent, for each action
                n_agents = self.num_agents
                n_actions = self.action_space
                # Graph Code
                StopRunning = False
                if n_agents == 3:
                    fig = plt.figure(figsize=(16, 12))
                    bottomValue = 0.54
                elif n_agents == 4:
                    fig = plt.figure(figsize=(16, 16))
                    bottomValue = 0.54 * (12 / 16)
                else:
                    print("Too many agents to display")
                    break

                # Game Render Graph is top left
                GameRenderAx = fig.add_axes([0.03, bottomValue, 0.48, 0.42], zorder=1)

                # Social Influence Graph is top right
                socialInfluence_ax = fig.add_axes(
                    [0.53, bottomValue + 0.04, 0.42, 0.38], zorder=1
                )

                # Graph type annotation stays at same place - just made smaller
                GraphTypeax = fig.add_axes(
                    [0.03, bottomValue + 0.3, 0.12, 0.12], zorder=2
                )  # Bigger zorder to be on top of the other two

                CurrentFrame = AddRenderGraph[
                    count + 2
                ]  # Because there is an excessive first frame and we are showing the next frame where actions are done
                GameRenderAx.imshow(CurrentFrame)
                GameRenderAx.axis("off")
                # Scale
                # GameRenderAx.set_aspect('auto')
                GameRenderAx.set_aspect(1)
                GameRenderAx.set_facecolor("white")
                GameRenderAx.set_title("Game")

                # InfoAx is just a text field
                # InfoAx.axis('off')
                GraphTypeax.set_facecolor((1, 1, 1, 0.75))

                FirstRightSideBlock = True

                if count + 1 < len(ListByTimestep):

                    #### Making the bottom sub plots - Do by agent

                    Height = 0.01 + bottomValue
                    TopPadding = 0.003
                    HeightPadding = 2 * TopPadding * (n_agents - 1) + 0.05
                    PerUnitHeight = ((Height - HeightPadding) / n_agents) - TopPadding
                    WidthOfBigPlot = 0.09
                    WidthOfSmallPlot = 0.08  # 20 / 10
                    SidePadding = 0.01
                    LabelSize = 0.04
                    MiniLabelSize = 0.005
                    ShortBlockSize = PerUnitHeight / 2 - (2 * MiniLabelSize)
                    WidthOfText = 0.06
                    SpaceForText = WidthOfText + SidePadding
                    RightSideBlockLocLeft = WidthOfBigPlot + 0.02 + SpaceForText

                    # SpaceForText = WidthOfBigPlot+0.05 + WidthOfSmallPlot

                    aliveMaskTimestep_SI = aliveMask[
                        count + 1, :, :
                    ]  # We want to know if dead agent for this timestep.
                    aliveMaskTimestep_flat_SI = aliveMaskTimestep_SI.flatten()

                    AliveAgentsList = list(range(n_agents))
                    # Remove dead agents according to the alive mask (aliveMaskTimestep)
                    AliveAgentsList = [
                        i for i in AliveAgentsList if aliveMaskTimestep_flat_SI[i] == 1
                    ]

                    for agent in AliveAgentsList:
                        # MyHeightFromBottom = Height - (agent * PerUnitHeight +  (2 * TopPadding) )
                        MyHeightFromBottom = (
                            Height
                            - (2 * TopPadding)
                            - (agent * (PerUnitHeight + TopPadding))
                            - PerUnitHeight / 2
                            - 0.1
                        )  # - (LabelSize * agent)
                        if n_agents > 3:
                            extraAgents = n_agents - 3
                            MyHeightFromBottom -= PerUnitHeight * (extraAgents * agent)
                        # print(MyHeightFromBottom)
                        if MyHeightFromBottom < 0:
                            MyHeightFromBottom = 0  # This is a catch that will just print it at the bottom even if it overlaps
                        LeftSideBlock = fig.add_axes(
                            [
                                0.02,
                                MyHeightFromBottom + TopPadding + LabelSize,
                                WidthOfBigPlot,
                                PerUnitHeight - (LabelSize * 1.5),
                            ],
                            zorder=1,
                        )
                        # RightSideBlock = fig.add_axes([0.92, MyHeightFromBottom, WidthOfBigPlot, PerUnitHeight - LabelSize], zorder=1)

                        LeftSideBlock.set_facecolor("gray")
                        LeftSideBlock.set_xticks([])
                        LeftSideBlock.set_yticks([])

                        # Add a number to the block
                        # wrapped_text = textwrap.fill(f"True Policy for Agent {agent}, for true actions", width=20)
                        # LeftSideBlock.text(0.5, 0.5, wrapped_text, horizontalalignment='center', verticalalignment='center', transform=LeftSideBlock.transAxes, fontsize=8, color='white')
                        # wrapped_text = textwrap.fill(f"Predicted Policy for Agent {agent}, for true actions", width=20)
                        # RightSideBlock.text(0.5, 0.5, wrapped_text, horizontalalignment='center', verticalalignment='center', transform=RightSideBlock.transAxes, fontsize=8, color='white')

                        # Check for nested list

                        policiesTS = signal_process_remove_low_probs(policiesTS)
                        next_policiesTS = signal_process_remove_low_probs(
                            next_policiesTS
                        )

                        # MyPolicy = policiesTS[agent]
                        # TrueActionTaken = actionsTS[agent]

                        # print("My policy size", MyPolicy.shape)
                        # print("True action taken size", TrueActionTaken.shape)

                        # print("Len List - MyPolicy", len(MyPolicy))
                        # print("Len Actions - TrueAction", len(TrueActionTaken))

                        # policies for next timestep
                        TrueActionNextTS = next_actionsTS[agent]
                        MyPolicyNextTS = next_policiesTS[agent]
                        LeftSideBlock = self.plotPolicyIntoHistogram(
                            LeftSideBlock,
                            policy=MyPolicyNextTS,
                            action=TrueActionNextTS,
                            agent_id=agent,
                            plotXLabelsLong=False,
                            titles=True,
                        )
                        LeftSideBlock.set_title(
                            f"Agent {agent} True Policy", fontsize=8
                        )

                        # Add title of Agent X
                        # LeftSideBlock.text(0.5, 0.9, f"Agent {agent}'s true policy in t{count+1}", horizontalalignment='center', verticalalignment='center', transform=LeftSideBlock.transAxes, fontsize=8, color='black')

                        # latex_string = fr"$\pi^{{A {{{agent}}} , T{{{count+1}}}}}$"
                        latex_string = rf"$\pi^{{{{{count+1}}}}}_{{{{{agent}}}}}$"

                        # print(f"Latex Printing - expect this to take a while TS{count}")
                        LeftSideBlock.text(
                            0.05,
                            0.88,
                            latex_string,
                            horizontalalignment="left",
                            verticalalignment="top",
                            transform=LeftSideBlock.transAxes,
                            fontsize=13,
                            color="black",
                            usetex=True,
                        )

                        OtherIDs = list(range(n_agents))
                        OtherIDs.remove(agent)

                        # Right side get predicted true action policies
                        for index in range(n_agents - 1):
                            OtherAgentID = OtherIDs[index]
                            if OtherAgentID in AliveAgentsList:
                                MySubCellHeight = (
                                    MyHeightFromBottom
                                    - (index * (PerUnitHeight + TopPadding) / 2)
                                    + (PerUnitHeight / 2)
                                )
                                # Text
                                LeftSideTextBlock = fig.add_axes(
                                    [
                                        WidthOfBigPlot + 0.025,
                                        MySubCellHeight,
                                        WidthOfText,
                                        ShortBlockSize,
                                    ],
                                    zorder=1,
                                )
                                # RightSideBlock = fig.add_axes([0.92, MyHeightFromBottom, WidthOfBigPlot, PerUnitHeight - LabelSize], zorder=1)

                                LeftSideTextBlock.set_facecolor("white")
                                LeftSideTextBlock.set_xticks([])
                                LeftSideTextBlock.set_yticks([])
                                # remove frame
                                LeftSideTextBlock.set_axis_off()

                                # info_string = f"Agent {OtherAgentID}'s predictions for Agent {agent}'s policy in t{count+1} if Agent {OtherAgentID} does other actions in t{count}"
                                info_string = f"Agent {OtherAgentID}'s predictions."
                                wrapped_info_string = textwrap.fill(
                                    info_string, width=15
                                )

                                LeftSideTextBlock.text(
                                    0.5,
                                    0.5,
                                    wrapped_info_string,
                                    fontsize=10,
                                    ha="center",
                                )

                                ############# True action policy
                                OtherAgentIDPastAction = actionsTS[OtherAgentID]

                                Block = fig.add_axes(
                                    [
                                        RightSideBlockLocLeft,
                                        MySubCellHeight,
                                        WidthOfSmallPlot,
                                        ShortBlockSize,
                                    ],
                                    zorder=1,
                                )
                                Block.set_facecolor("lightgray")
                                Block.set_xticks([])
                                Block.set_yticks([])

                                _, MySIPolicy = (
                                    self.select_social_influence_for_predicted_policies(
                                        CounterFactualInfo,
                                        count,
                                        OtherAgentID,
                                        OtherAgentIDPastAction,
                                        agent,
                                        TrueActionUsed=OtherAgentIDPastAction,
                                        WhetherTrueAction=True,
                                    )
                                )

                                if MySIPolicy is not None:
                                    hideTicks = (
                                        index != n_agents - 2 or agent != n_agents - 1
                                    )
                                    Block = self.plotPolicyIntoHistogram(
                                        Block,
                                        policy=MySIPolicy,
                                        action=OtherAgentIDPastAction,
                                        agent_id=OtherAgentID,
                                        plotXLabelsLong=False,
                                        titles=False,
                                        hideTicks=hideTicks,
                                    )

                                    # Block.text(0.5, 0.9, f"Agent {agent}'s predicted policy in t{count+1} if {OtherAgentID} does {OtherAgentIDPastAction} t{count}", horizontalalignment='center', verticalalignment='center', transform=LeftSideBlock.transAxes, fontsize=8, color='black')

                                    OtherAgentIDPastAction_name = self.actionTranslater(
                                        OtherAgentIDPastAction
                                    )

                                    # latex_string = fr"$\tilde{{\pi}}_{{A{{{OtherAgentID}}}, ({{{OtherAgentIDPastAction_name}}}), T{{{count}}}}}^{{A{{{agent}}}, T{{{count+1}}}}}$"
                                    # latex_string = fr"$\tilde{{\pi}}_{{{{{agent}}},{{{OtherAgentID}}}}}^{{{{{count+1}}}}} ({{{OtherAgentIDPastAction_name}}})$"
                                    latex_string = rf"$\tilde{{\pi}}^{{{count+1}}}_{{{agent}}} ({{{OtherAgentIDPastAction_name}}}^{{{count}}}_{{{OtherAgentID}}})$"

                                    # print("Latex Printing Block")

                                    Block.text(
                                        0.05,
                                        0.88,
                                        latex_string,
                                        horizontalalignment="left",
                                        verticalalignment="top",
                                        transform=Block.transAxes,
                                        fontsize=12,
                                        color="black",
                                        usetex=True,
                                    )
                                    if FirstRightSideBlock:
                                        # Add a title
                                        Block.set_title(
                                            "Predicted Policies for True Action",
                                            fontsize=8,
                                        )

                                    FirstRightSideBlock = False

                        for other_agent in range(n_agents - 1):
                            MySubCellHeight = (
                                MyHeightFromBottom
                                - (other_agent * ((PerUnitHeight + TopPadding) / 2))
                                + (PerUnitHeight / 2)
                            )
                            # print("other agent: ", other_agent, "MySubCellHeight: ", MySubCellHeight)

                            OtherAgentID = OtherIDs[other_agent]

                            if OtherAgentID in AliveAgentsList:
                                lastRow = (
                                    agent == n_agents - 1
                                    and other_agent == n_agents - 2
                                )
                                actionSet = list(
                                    range(1, n_actions)
                                )  # DO WE WANT TO PREDICT IF I WAS DEAD? but not trained
                                OtherAgentIDPastAction = actionsTS[OtherAgentID]
                                # remove the true action
                                # actionSet.remove(OtherAgentIDPastAction)
                                for leftIndex, counterfacutal_action in enumerate(
                                    actionSet
                                ):  # We don't need for no op
                                    # FirstRightCounterBlock = agent == 0 and other_agent == 0 and leftIndex == math.floor(len(actionSet)/2)
                                    TopRow = agent == 0 and other_agent == 0

                                    # Original side
                                    MyLeftLoc = (
                                        RightSideBlockLocLeft
                                        + (2 * SidePadding)
                                        + (
                                            (leftIndex)
                                            * (WidthOfSmallPlot + SidePadding)
                                        )
                                        + WidthOfSmallPlot
                                    )
                                    # print("Action Counter: ", Index, "MyLeftLoc: ", MyLeftLoc)

                                    # SmallBlock.set_facecolor('blue')

                                    # SmallBlock.set_xticks([])
                                    # SmallBlock.set_yticks([])
                                    # Add a number to the block
                                    # wrapped_text = textwrap.fill(f"Alt Policy for Agent {agent}, if Agent {OtherAgentID} does {counterfacutal_action}", width=20)
                                    # SmallBlock.text(0.5, 0.5, wrapped_text, horizontalalignment='center', verticalalignment='center', transform=SmallBlock.transAxes, fontsize=8, color='white')

                                    # randomJS = np.random.rand()
                                    # randomPolicy = np.random.rand(9)
                                    # # normalize to 1
                                    # randomPolicy = [x / sum(randomPolicy) for x in randomPolicy]

                                    SmallBlock = fig.add_axes(
                                        [
                                            MyLeftLoc,
                                            MySubCellHeight,
                                            WidthOfSmallPlot,
                                            ShortBlockSize,
                                        ],
                                        zorder=1,
                                    )

                                    if OtherAgentIDPastAction != counterfacutal_action:

                                        MySI, MySIPolicy = (
                                            self.select_social_influence_for_predicted_policies(
                                                CounterFactualInfo,
                                                count,
                                                OtherAgentID,
                                                counterfacutal_action,
                                                agent,
                                                TrueActionUsed=OtherAgentIDPastAction,
                                                WhetherTrueAction=False,
                                            )
                                        )

                                        if MySIPolicy is not None:
                                            # counterfactual_action = counterfacutal_action
                                            # if counterfacutal_action >= OtherAgentIDPastAction:
                                            #     counterfactual_action += 1

                                            SmallBlock = self.PlotSmallPlots(
                                                SmallBlock,
                                                policy=MySIPolicy,
                                                JS_Div=MySI,
                                                agent_predicting=OtherAgentID,
                                                counterfactual_action=counterfacutal_action,
                                                lastRow=lastRow,
                                                agent_predicted_for=agent,
                                                current_time=count,
                                            )

                                        else:
                                            # Make this a blank block
                                            SmallBlock.set_facecolor("white")
                                            SmallBlock.set_xticks([])
                                            SmallBlock.set_yticks([])
                                            SmallBlock.set_axis_off()
                                    else:
                                        # Make this a blank block
                                        SmallBlock.set_facecolor("white")
                                        SmallBlock.set_xticks([])
                                        SmallBlock.set_yticks([])
                                        SmallBlock.set_axis_off()

                                    # if FirstRightCounterBlock:
                                    #     #Add a title

                                    #     SmallBlock.set_title(f"Predicted Policies in timestep {count+1} for Counterfactual Actions taken in timestep {count}", fontsize=8)

                                    if TopRow:
                                        # Add a title
                                        counterfacutal_action_name = (
                                            self.actionTranslater(counterfacutal_action)
                                        )
                                        SmallBlock.set_title(
                                            counterfacutal_action_name, fontsize=14
                                        )

            else:
                if AddRenderGraph is not None and count + 2 < len(AddRenderGraph):
                    fig = plt.figure(figsize=(16, 12))
                    # uses rows and columns so actually y x
                    # InfoAx = plt.subplot2grid((3, 4), (0, 0), colspan=1, rowspan=1)
                    # GameRenderAx = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3)
                    # GraphTypeax = plt.subplot2grid((3, 4), (0, 0), colspan=1, rowspan=1)
                    # socialInfluence_ax = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=3)
                    if False:
                        GameRenderAx = fig.add_axes([0.05, 0.05, 0.40, 0.80], zorder=1)
                        socialInfluence_ax = fig.add_axes(
                            [0.5, 0.05, 0.45, 0.80], zorder=1
                        )
                    else:  # New version of scaling graph because of probability on the other side
                        GameRenderAx = fig.add_axes([0.01, 0.05, 0.47, 0.80], zorder=1)
                        socialInfluence_ax = fig.add_axes(
                            [0.5, 0.05, 0.45, 0.80], zorder=1
                        )

                    GraphTypeax = fig.add_axes(
                        [0.01, 0.6, 0.25, 0.25], zorder=2
                    )  # Bigger zorder to be on top of the other two
                    ## Display the render graph
                    CurrentFrame = AddRenderGraph[count + 2]
                    GameRenderAx.imshow(CurrentFrame)
                    GameRenderAx.axis("off")
                    # Scale
                    # GameRenderAx.set_aspect('auto')
                    GameRenderAx.set_aspect(1)
                    GameRenderAx.set_facecolor("white")
                    GameRenderAx.set_title("Game")

                    # InfoAx is just a text field
                    # InfoAx.axis('off')
                    GraphTypeax.set_facecolor((1, 1, 1, 0.7))

                else:
                    fig = plt.figure(figsize=(16, 12))
                    # Super impose on top right https://stackoverflow.com/questions/44678878/superimpose-independent-plots-in-python
                    socialInfluence_ax = plt.subplot2grid(
                        (3, 3), (0, 0), colspan=2, rowspan=3
                    )  # Takes up 2/3 of width and full height
                    GraphTypeax = plt.subplot2grid(
                        (3, 3), (0, 2), rowspan=3
                    )  # Takes up 1/3 of width and full height
                    GraphTypeax.set_facecolor("white")

                GraphTypeax.set_xticks([])
                GraphTypeax.set_yticks([])

            if ShouldIPlotSI:

                GraphTypeax.set_aspect(1.2)
                GraphTypeax = self.si_graph_checker.draw_graph_type(
                    GraphType,
                    GraphTypeax,
                    CustomParams={"node_size": 400, "arrowsize": 10, "width": 2},
                )

                #### Draw the graph
                nx.draw(
                    G,
                    ax=socialInfluence_ax,
                    with_labels=True,
                    font_weight="bold",
                    pos=node_positions,
                    node_size=1200,
                    arrowsize=50,
                    font_size=14,
                    font_color="black",
                    edge_color=colorsEdges,
                    width=5,
                    edge_cmap=plt.cm.Blues,
                    arrows=True,
                    connectionstyle="arc3, rad = 0.15",
                )
                nx.draw_networkx_edge_labels(
                    G,
                    ax=socialInfluence_ax,
                    pos=node_positions,
                    edge_labels=nx.get_edge_attributes(G, "weight"),
                    font_color="black",
                    font_size=14,
                    label_pos=0.3,
                )

                if len(weights) > 0:
                    # Create a new axes for the colorbar
                    divider = make_axes_locatable(socialInfluence_ax)
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)

                    minimum = min(list(weights))
                    maximum = max(list(weights))

                    sm = plt.cm.ScalarMappable(
                        cmap=plt.cm.plasma,
                        norm=colors.Normalize(vmin=minimum, vmax=maximum),
                    )

                    # Create a colorbar
                    # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=colors.Normalize(vmin=min(list(weights)), vmax=max(list(weights))))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")

                    # Set the colorbar ticks
                    cbar.set_ticks(np.unique(weights))

                    # Set the colorbar tick labels
                    cbar.set_ticklabels(np.unique(weights))

                    # Adjust size of colorbar
                    cbar.ax.tick_params(labelsize=8)
                    cbar.set_label(
                        "Social Influence", rotation=0, labelpad=10, fontsize=10
                    )
            else:
                GraphTypeax.set_axis_off()
                socialInfluence_ax.set_axis_off()

            UsedCount = count + 2
            rollout = UsedCount // self.episode_length
            ts = UsedCount % self.episode_length
            if ShouldIPlotSI:
                socialInfluence_ax.set_title(
                    f"Timestep {ts-1} - Pruned below {PercentilePrune}th percentile (Threshold: {NameThreshold})"
                )

            # Name

            fileName = f"{self.gifs_dir}/P-Ep{episode}_R{rollout}_TS{ts}-social_influence_graph.png"

            # if AddRenderGraph is not None:
            #     fig.title(0, 0, f"Timestep {ts} - Pruned below {PercentilePrune}th percentile (Threshold: {NameThreshold})", ha='left', va='top', fontsize=14, wrap=True)
            # else:

            # plt.show()
            fig.savefig(fileName)
            CurrentRollOut.append(fileName)

            # clear the graph
            plt.clf()
            plt.cla()
            plt.close("all")

            # plt.close(fig)

            # if this is the last timestep for this rollout, make a gif
            # if (count + 1) // self.episode_length != rollout:
            if count + 1 == NumberOfTimesteps:
                gif_save_path = self.make_gif_from_images(
                    episode, rollout, CurrentRollOut, evaluation
                )
                CurrentRollOut = []
                GraphProduced = True

        if not GraphProduced:
            if len(CurrentRollOut) == 0:
                print("No images to make gif from but this graph shape", graph.shape)
            else:
                # Max episode length not used since we are not using the rollout, single episode evaluation
                gif_save_path = self.make_gif_from_images(
                    episode, "-SingleEval", CurrentRollOut, evaluation
                )

        return gif_save_path  # only the last one because we just need one

    def save_twin_social_influence_diagram_pruning(
        self,
        graph,
        aliveMask,
        positionOfAgents,
        episode,
        evaluation,
        PercentilePrune=35,
        AddRenderGraph=False,
        BigInfoOfSIPolicies=None,
    ):

        if BigInfoOfSIPolicies[0] is not None:
            raise NotImplementedError(
                "This function is not implemented for Twin Social Influence Diagrams"
            )

        # plot game and two SI side by side
        numberOfOtherAgents = self.num_agents - 1
        gif_save_path = None
        GraGraphProduced = False
        CurrentRollOut = []
        rollout = 0
        count = 0

        A_Graph, B_Graph = graph

        assert A_Graph.shape[0] == self.num_agents, "shape check for A_Graph"
        assert A_Graph.shape[1] == self.num_agents, "shape check for A_Graph"

        assert B_Graph.shape[0] == self.num_agents, "shape check for B_Graph"
        assert B_Graph.shape[1] == self.num_agents, "shape check for B_Graph"

        ListByTimestepA = unstack(A_Graph, 2)
        ListByTimestepB = unstack(B_Graph, 2)

        DisplayListByTimestepA, TwentyFifthPercentileA, NameThresholdA, MaxA, MinA = (
            self.graph_filtering(A_Graph, PercentilePrune)
        )
        DisplayListByTimestepB, TwentyFifthPercentileB, NameThresholdB, MaxB, MinB = (
            self.graph_filtering(B_Graph, PercentilePrune)
        )

        # sharedMax
        MaxSIForAllTimeSteps = max(MaxA, MaxB)
        MinSIForAllTimeSteps = min(MinA, MinB)

        TotalTimesteps = len(ListByTimestepA)
        NumberOfTimesteps = (
            len(ListByTimestepA) - 2
        )  # We are plotting the next timestep for SI

        bottomValue = 0.02  # for the graphs
        LeftMostValue = 0.01
        ThirdSectionLeft = 0.68

        for count in range(-2, NumberOfTimesteps):
            ShouldIPlotSI = count >= 0 and count + 1 < TotalTimesteps
            if ShouldIPlotSI:
                ###### Consider two teams
                timestep_graphA = ListByTimestepA[count + 1]
                display_timestep_graphA = DisplayListByTimestepA[count + 1]
                timestep_graphB = ListByTimestepB[count + 1]
                display_timestep_graphB = DisplayListByTimestepB[count + 1]

                aliveMaskTimestep = aliveMask[count + 1, :, :]
                aliveMaskTimestep_flat = aliveMaskTimestep.flatten()
                numberLivingAgents = int(np.sum(aliveMaskTimestep))
                assert (
                    numberLivingAgents <= self.num_agents
                ), "All agents should be less than the total number of agents"

                if numberLivingAgents >= 2:
                    # Assuming node_labels is your tensor containing node labels
                    # and aliveMaskTimestep is a tensor of the same shape containing boolean values
                    AliveAgentnode = np.where(aliveMaskTimestep == 1)[0]
                    ###### Consider two teams
                    G_a = nx.DiGraph()
                    G_b = nx.DiGraph()

                    G_a.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes
                    G_b.add_nodes_from(range(numberLivingAgents))

                    living_agents = [
                        i
                        for i in range(self.num_agents)
                        if aliveMaskTimestep_flat[i] == 1
                    ]

                    for cx, i in enumerate(living_agents):
                        # add number to agent nodes
                        G_a.nodes[cx]["label"] = i
                        G_b.nodes[cx]["label"] = i
                        for cy, j in enumerate(living_agents):

                            weightA = timestep_graphA[i][j]
                            display_weightA = display_timestep_graphA[i][j]

                            weightB = timestep_graphB[i][j]
                            display_weightB = display_timestep_graphB[i][j]

                            # team a
                            if (
                                i != j
                                and weightA != 0
                                and weightA > 1e-8
                                and weightA > TwentyFifthPercentileA
                            ):
                                NewW = round(display_weightA, 2)
                                if NewW == 0:
                                    NewW = round(display_weightA, 3)
                                if NewW != 0:
                                    G_a.add_weighted_edges_from([(cx, cy, NewW)])

                            # team b
                            if (
                                i != j
                                and weightB != 0
                                and weightB > 1e-8
                                and weightB > TwentyFifthPercentileB
                            ):
                                NewW = round(display_weightB, 2)
                                if NewW == 0:
                                    NewW = round(display_weightB, 3)
                                if NewW != 0:
                                    G_b.add_weighted_edges_from([(cx, cy, NewW)])

                    # Save fixed positins of nodes ------ just need to do for one graph and can share
                    node_positions = nx.circular_layout(G_a)
                    if positionOfAgents is not None:
                        # node_positions = nx.circular_layout(G) # Use this in case missing an agent
                        positionOfAgentsThisTime = positionOfAgents[count, :, :]
                        # only get living agents positions - delete the dead agents
                        filteredPositionOfAgentsThisTime = positionOfAgentsThisTime[
                            AliveAgentnode
                        ]  # (numberLivingAgents, 2)
                        # print("filteredPositionOfAgentsThisTime", filteredPositionOfAgentsThisTime.shape)
                        # make into format of networkx
                        node_positions.update(
                            {
                                i: (
                                    filteredPositionOfAgentsThisTime[i][0],
                                    filteredPositionOfAgentsThisTime[i][1],
                                )
                                for i in range(numberLivingAgents)
                            }
                        )

                    else:
                        if self.graph_node_positions is None:
                            # node_positions = nx.circular_layout(G)
                            self.graph_node_positions = node_positions
                        else:
                            node_positions = self.graph_node_positions

                    ## Do some adjusting of positions for better visibility
                    node_positions = nx.spring_layout(
                        G_a, pos=node_positions, iterations=6
                    )

                    #### Color of edges ---- make these shared for both graphs
                    # Get the weights of the edges
                    weights_A = nx.get_edge_attributes(G_a, "weight").values()
                    weights_B = nx.get_edge_attributes(G_b, "weight").values()

                    # Normalize the weights to the range [0, 1]
                    ComboWeights = list(weights_A) + list(weights_B)
                    weights = np.array(ComboWeights)

                    weights_normalized = (weights - MinSIForAllTimeSteps) / (
                        MaxSIForAllTimeSteps - MinSIForAllTimeSteps
                    )

                    colorsEdges = plt.cm.plasma(weights_normalized)

                #### Subplot to show graph type
                GraphType_A = self.si_graph_checker.check_which_graph(G_a)
                GraphType_B = self.si_graph_checker.check_which_graph(G_b)

                if (
                    AddRenderGraph is not None
                    and count + 2 < len(AddRenderGraph)
                    and count >= 0
                ):
                    # Making the simple version of graph with SI diagrams
                    n_agents = self.num_agents
                    n_actions = self.action_space
                    # Graph Code
                    fig = plt.figure(figsize=(12, 4))

                    # Left Bottom Width Height

                    # Game Render Graph is top left
                    GameRenderAx = fig.add_axes(
                        [0.34, bottomValue, 0.30, 0.94], zorder=1
                    )

                    # Social Influence Graph is top right
                    socialInfluence_ax_A = fig.add_axes(
                        [0.02, bottomValue + 0.04, 0.30, 0.75], zorder=1
                    )
                    socialInfluence_ax_B = fig.add_axes(
                        [ThirdSectionLeft, bottomValue + 0.04, 0.30, 0.75], zorder=1
                    )

                    # Graph type annotation stays at same place - just made smaller
                    GraphTypeax_A = fig.add_axes(
                        [LeftMostValue - 0.015, 0.8, 0.12, 0.20], zorder=2
                    )  # Bigger zorder to be on top of the other two
                    GraphTypeax_B = fig.add_axes(
                        [ThirdSectionLeft - 0.03, 0.8, 0.12, 0.20], zorder=2
                    )  # Bigger zorder to be on top of the other two

                    CurrentFrame = AddRenderGraph[
                        count + 2
                    ]  # Because there is an excessive first frame and we are showing the next frame where actions are done
                    GameRenderAx.imshow(CurrentFrame)
                    GameRenderAx.axis("off")
                    GameRenderAx.set_aspect(1)
                    GameRenderAx.set_facecolor("white")
                    GameRenderAx.set_title("Game")

                    socialInfluence_ax_A.set_title("Social Influence Team A")
                    socialInfluence_ax_B.set_title("Social Influence Team B")

                    # InfoAx is just a text field
                    # InfoAx.axis('off')
                    GraphTypeax_A.set_facecolor((1, 1, 1, 0.75))
                    GraphTypeax_B.set_facecolor((1, 1, 1, 0.75))

            else:
                if AddRenderGraph is not None and count + 2 < len(AddRenderGraph):
                    fig = plt.figure(figsize=(12, 4))

                    ## Display the render graph
                    CurrentFrame = AddRenderGraph[count + 2]

                    GameRenderAx = fig.add_axes(
                        [0.34, bottomValue, 0.30, 0.94], zorder=1
                    )

                    # Social Influence Graph is top right
                    socialInfluence_ax_A = fig.add_axes(
                        [0.01 + LeftMostValue, bottomValue + 0.04, 0.30, 0.75], zorder=1
                    )
                    socialInfluence_ax_B = fig.add_axes(
                        [ThirdSectionLeft, bottomValue + 0.04, 0.30, 0.75], zorder=1
                    )

                    # Graph type annotation stays at same place - just made smaller
                    GraphTypeax_A = fig.add_axes(
                        [LeftMostValue - 0.015, 0.8, 0.12, 0.20], zorder=2
                    )  # Bigger zorder to be on top of the other two
                    GraphTypeax_B = fig.add_axes(
                        [ThirdSectionLeft - 0.03, 0.8, 0.12, 0.20], zorder=2
                    )  # Bigger zorder to be on top of the other two

                    GameRenderAx.imshow(CurrentFrame)
                    GameRenderAx.axis("off")
                    # Scale
                    # GameRenderAx.set_aspect('auto')
                    GameRenderAx.set_aspect(1)
                    GameRenderAx.set_facecolor("white")
                    GameRenderAx.set_title("Game")

                    # InfoAx is just a text field
                    # InfoAx.axis('off')
                    GraphTypeax_A.set_facecolor((1, 1, 1, 0.7))
                    GraphTypeax_B.set_facecolor((1, 1, 1, 0.7))

                else:
                    fig = plt.figure(figsize=(12, 4))
                    # Super impose on top right https://stackoverflow.com/questions/44678878/superimpose-independent-plots-in-python
                    # Left plot for Social Influence Team A
                    socialInfluence_ax_A = plt.subplot2grid(
                        (3, 3), (0, 0), colspan=1, rowspan=3
                    )  # Takes up 1/3 of width and full height

                    # Right plot for Social Influence Team B
                    socialInfluence_ax_B = plt.subplot2grid(
                        (3, 3), (0, 2), colspan=1, rowspan=3
                    )  # Takes up 1/3 of width and full height

                    # Center plot for the game
                    GameRenderAx = plt.subplot2grid(
                        (3, 3), (0, 1), colspan=1, rowspan=3
                    )  # Takes up 1/3 of width and full height

                    # Set face colors for the axes
                    socialInfluence_ax_A.set_facecolor("white")
                    socialInfluence_ax_B.set_facecolor("white")
                    GraphTypeax_A = plt.subplot2grid(
                        (3, 3), (0, 2), rowspan=3
                    )  # Takes up 1/3 of width and full height
                    GraphTypeax_A.set_facecolor("white")
                    GraphTypeax_B = plt.subplot2grid(
                        (3, 3), (0, 2), rowspan=3
                    )  # Takes up 1/3 of width and full height
                    GraphTypeax_B.set_facecolor("white")

                GraphTypeax_A.set_xticks([])
                GraphTypeax_A.set_yticks([])
                GraphTypeax_B.set_xticks([])
                GraphTypeax_B.set_yticks([])

            if ShouldIPlotSI:
                GraphTypeax_A.set_aspect(1.1)
                GraphTypeax_B.set_aspect(1.1)
                GraphTypeNodeSize = 40
                GraphTypeArrowSize = 3
                GraphTypeAdjustedWidth = 1
                GraphTypeax_A = self.si_graph_checker.draw_graph_type(
                    GraphType_A,
                    GraphTypeax_A,
                    CustomParams={
                        "node_size": GraphTypeNodeSize,
                        "arrowsize": GraphTypeArrowSize,
                        "width": GraphTypeAdjustedWidth,
                    },
                )
                GraphTypeax_B.set_aspect(1.2)
                GraphTypeax_B = self.si_graph_checker.draw_graph_type(
                    GraphType_B,
                    GraphTypeax_B,
                    CustomParams={
                        "node_size": GraphTypeNodeSize,
                        "arrowsize": GraphTypeArrowSize,
                        "width": GraphTypeAdjustedWidth,
                    },
                )

                AdjustedNodeSize = 500
                AdjustedArrowSize = 20
                fontSize = 10
                AdjustedWidth = 3

                #### Draw the graph
                nx.draw(
                    G_a,
                    ax=socialInfluence_ax_A,
                    with_labels=True,
                    font_weight="bold",
                    pos=node_positions,
                    node_size=AdjustedNodeSize,
                    arrowsize=AdjustedArrowSize,
                    font_size=14,
                    font_color="black",
                    edge_color=colorsEdges,
                    width=AdjustedWidth,
                    edge_cmap=plt.cm.Blues,
                    arrows=True,
                    connectionstyle="arc3, rad = 0.15",
                )
                nx.draw_networkx_edge_labels(
                    G_a,
                    ax=socialInfluence_ax_A,
                    pos=node_positions,
                    edge_labels=nx.get_edge_attributes(G_a, "weight"),
                    font_color="black",
                    font_size=fontSize,
                    label_pos=0.3,
                )
                #
                nx.draw(
                    G_b,
                    ax=socialInfluence_ax_B,
                    with_labels=True,
                    font_weight="bold",
                    pos=node_positions,
                    node_size=AdjustedNodeSize,
                    arrowsize=AdjustedArrowSize,
                    font_size=14,
                    font_color="black",
                    edge_color=colorsEdges,
                    width=AdjustedWidth,
                    edge_cmap=plt.cm.Blues,
                    arrows=True,
                    connectionstyle="arc3, rad = 0.15",
                )
                nx.draw_networkx_edge_labels(
                    G_b,
                    ax=socialInfluence_ax_B,
                    pos=node_positions,
                    edge_labels=nx.get_edge_attributes(G_b, "weight"),
                    font_color="black",
                    font_size=fontSize,
                    label_pos=0.3,
                )
                #
                # nx.draw(G,ax=socialInfluence_ax, with_labels=True, font_weight='bold', pos = node_positions, node_size=1200, arrowsize=50, font_size=14, font_color='black', edge_color=colorsEdges, width=5, edge_cmap=plt.cm.Blues, arrows=True, connectionstyle='arc3, rad = 0.15')
                # nx.draw_networkx_edge_labels(G,ax=socialInfluence_ax, pos=node_positions, edge_labels=nx.get_edge_attributes(G, 'weight'), font_color='black', font_size=14, label_pos=0.3)

                if len(weights) > 0:
                    # Create a new axes for the colorbar
                    dividerA = make_axes_locatable(socialInfluence_ax_A)
                    dividerB = make_axes_locatable(socialInfluence_ax_B)
                    cax_A = dividerA.append_axes("bottom", size="5%", pad=0.05)
                    cax_B = dividerB.append_axes("bottom", size="5%", pad=0.05)

                    minimum = min(list(weights))
                    maximum = max(list(weights))

                    sm = plt.cm.ScalarMappable(
                        cmap=plt.cm.plasma,
                        norm=colors.Normalize(vmin=minimum, vmax=maximum),
                    )

                    # Create a colorbar
                    # sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=colors.Normalize(vmin=min(list(weights)), vmax=max(list(weights))))
                    sm.set_array([])
                    cbar_A = plt.colorbar(sm, cax=cax_A, orientation="horizontal")
                    cbar_B = plt.colorbar(sm, cax=cax_B, orientation="horizontal")

                    # Set the colorbar ticks
                    # cbar.set_ticks(np.unique(weights))
                    cbar_A.set_ticks(np.unique(weights))
                    cbar_B.set_ticks(np.unique(weights))

                    # Set the colorbar tick labels
                    # cbar.set_ticklabels(np.unique(weights))
                    cbar_A.set_ticklabels(np.unique(weights))
                    cbar_B.set_ticklabels(np.unique(weights))

                    # Adjust size of colorbar
                    # cbar.ax.tick_params(labelsize=8)
                    # cbar.set_label('Social Influence', rotation=0, labelpad=10, fontsize=10)
                    cbar_A.ax.tick_params(labelsize=8)
                    cbar_A.set_label(
                        "Social Influence", rotation=0, labelpad=10, fontsize=10
                    )
                    cbar_B.ax.tick_params(labelsize=8)
                    cbar_B.set_label(
                        "Social Influence", rotation=0, labelpad=10, fontsize=10
                    )
            else:
                GraphTypeax_A.set_axis_off()
                socialInfluence_ax_A.set_axis_off()
                GraphTypeax_B.set_axis_off()
                socialInfluence_ax_B.set_axis_off()

            UsedCount = count + 2
            rollout = UsedCount // self.episode_length
            ts = UsedCount % self.episode_length
            if ShouldIPlotSI:
                socialInfluence_ax_A.set_title(
                    f"(Team A) Timestep {ts-1} - {PercentilePrune}th Prune ({NameThresholdA:.3f})",
                    fontsize=8,
                    wrap=True,
                )
                socialInfluence_ax_B.set_title(
                    f"(Team B) Timestep {ts-1} - {PercentilePrune}th Prune ({NameThresholdB:.3f})",
                    fontsize=8,
                    wrap=True,
                )

            # Name

            fileName = f"{self.gifs_dir}/P-Ep{episode}_R{rollout}_TS{ts}-social_influence_graph.png"

            # if AddRenderGraph is not None:
            #     fig.title(0, 0, f"Timestep {ts} - Pruned below {PercentilePrune}th percentile (Threshold: {NameThreshold})", ha='left', va='top', fontsize=14, wrap=True)
            # else:

            # plt.show()
            fig.savefig(fileName)
            CurrentRollOut.append(fileName)

            # clear the graph
            plt.clf()
            plt.cla()
            plt.close("all")

            # plt.close(fig)

            # if this is the last timestep for this rollout, make a gif
            # if (count + 1) // self.episode_length != rollout:
            if count + 1 == NumberOfTimesteps:
                gif_save_path = self.make_gif_from_images(
                    episode, rollout, CurrentRollOut, evaluation
                )
                CurrentRollOut = []
                GraphProduced = True

        if not GraphProduced:
            if len(CurrentRollOut) == 0:
                print("No images to make gif from but this graph shape", graph.shape)
            else:
                # Max episode length not used since we are not using the rollout, single episode evaluation
                gif_save_path = self.make_gif_from_images(
                    episode, "-SingleEval", CurrentRollOut, evaluation
                )

        return gif_save_path  # only the last one because we just need one

    def graph_filtering(self, graph, PercentilePrune):
        flattenGraph = graph.flatten()
        # Remove all the zeroes
        flattenGraphCleaned = flattenGraph[flattenGraph > 1e-8]

        assert (
            sum(flattenGraphCleaned) > 0
        ), "There should be some social influence in the graph"

        TwentyFifthPercentile = np.percentile(
            flattenGraphCleaned, PercentilePrune
        )  # This is per episode

        NameThreshold = round(TwentyFifthPercentile, 4)
        ##### Clean up the weights for display
        DisplayGraph = np.where(graph < TwentyFifthPercentile, 0, graph)

        # Histogram equalization
        # DisplayGraph = np.where(DisplayGraph == 0, 0, np.log(DisplayGraph))
        ActivePoints = DisplayGraph != 0
        DisplayGraph = equalize_hist(DisplayGraph, nbins=256, mask=ActivePoints)

        DisplayListByTimestep = unstack(DisplayGraph, 2)

        Max = np.max(DisplayGraph)
        Min = np.min(DisplayGraph)

        return DisplayListByTimestep, TwentyFifthPercentile, NameThreshold, Max, Min

    def action_to_index(self, actionString):
        """Given an actionString, convert it to the index of the action"""
        return self.actionTypes.index(actionString)

    def graph_timestep_analysis(
        self,
        graph,
        aliveMask,
        episode=0,
        PercentilePrune=25,
        action_analysis=None,
        PlotHeatMap=False,
    ):

        flattenGraph = graph.flatten()
        # flattenGraphCleaned = flattenGraph
        # flattenGraphCleaned[flattenGraphCleaned <= 1e-8] = 0
        flattenGraphCleaned = flattenGraph[flattenGraph != 0]
        TwentyFifthPercentile = np.percentile(
            flattenGraphCleaned, PercentilePrune
        )  # This is per episode
        DisplayGraph = np.where(graph < TwentyFifthPercentile, 0, graph)

        MyGraphPerTimestep = []
        # This graph is checked for the type of graph it is at each timestep
        DisplayListByTimestep = unstack(DisplayGraph, 2)
        for count, timestep_graph in enumerate(DisplayListByTimestep):
            aliveMaskTimestep = aliveMask[count, :, :]
            aliveMaskTimestep_flat = aliveMaskTimestep.flatten()
            display_timestep_graph = DisplayListByTimestep[count]
            # print("aliveMaskTimestep", aliveMaskTimestep.shape)

            numberLivingAgents = int(np.sum(aliveMaskTimestep))
            assert (
                numberLivingAgents <= self.num_agents
            ), "All agents should be less than the total number of agents"

            if numberLivingAgents >= 2:
                # AliveAgentnode = np.where(aliveMaskTimestep == 1)[0]
                G = nx.DiGraph()
                G.add_nodes_from(range(numberLivingAgents))  # Add the agent nodes

                living_agents = [
                    i for i in range(self.num_agents) if aliveMaskTimestep_flat[i] == 1
                ]
                # print("Living agents", living_agents)
                for cx, i in enumerate(living_agents):
                    # add number to agent nodes
                    G.nodes[cx]["label"] = i
                    for cy, j in enumerate(living_agents):
                        weight = timestep_graph[i][j]
                        display_weight = display_timestep_graph[i][j]
                        # Prune the weights below a certain threshold (TwentyFifthPercentile)
                        if (
                            i != j
                            and weight != 0
                            and weight > 1e-8
                            and weight > TwentyFifthPercentile
                        ):
                            G.add_weighted_edges_from([(cx, cy, display_weight)])

                #### Subplot to show graph type
                GraphType = self.si_graph_checker.check_which_graph(G)
                MyGraphPerTimestep.append(GraphType)

        PreppedStrings = None
        if action_analysis is not None:
            # need to do one timestep because they action analysis is for the next timestep (actionAnalysis is a dictionary by timestep)
            actionsTaken = [
                action_analysis[i] for i in range(0, len(action_analysis.keys()))
            ]
            # actionsTaken = [action_analysis[i] for i in range(1, len(action_analysis.keys()))]
            # prepend = ["Start"] # prepend the first one with Nil
            # actionsTaken.insert(0, prepend)
            PreppedStrings = [
                ", ".join(info) if len(info) > 0 else "No Actions"
                for info in actionsTaken
            ]  # Sanity Check
            # print("number of actions originally ", len(action_analysis.keys())) #18
            # print("number of SI timesteps ", len(MyGraphPerTimestep)) #18

        Y_GraphPerTimestep = [
            self.si_graph_checker.get_score(a) for a in MyGraphPerTimestep
        ]
        ScoreLabels = {}
        # for tup, score in zip(MyGraphPerTimestep, Y_GraphPerTimestep):
        #     key, count = tup
        #     # title = f"Type {key}E-{count} {str(score)}"
        #     title = self.si_graph_checker.get_statement(tup)
        #     # title = self.si_graph_checker.statement[tup]
        #     ListOfLabels.append(title)

        Y_GraphPerTimestep = [score for score, _ in Y_GraphPerTimestep]
        # do a log of the score
        ScoreY = [np.log(score) if score > 0 else 0 for score in Y_GraphPerTimestep]

        for score, identification in zip(ScoreY, MyGraphPerTimestep):
            title = self.si_graph_checker.get_statement(identification)
            if ScoreLabels.get(score, None) is None:
                ScoreLabels[score] = [title]
            else:
                if title not in ScoreLabels[score]:
                    ScoreLabels[score].append(title)

        # Sort by score in key
        ScoreLabels = dict(sorted(ScoreLabels.items()))
        # make the lists into concat strings
        ListOfLabels = [
            ", ".join(value) + f" ({key})" for key, value in ScoreLabels.items()
        ]

        # style ggplot
        SavePath = None
        with plt.style.context("ggplot"):
            fig = plt.figure(figsize=(12, 8))
            if PreppedStrings is not None:
                # plot the line with no markers
                plt.plot(ScoreY, marker="")
                for x, y in enumerate(ScoreY):
                    InfoString = PreppedStrings[x]
                    ActionList = actionsTaken[x]
                    plt.plot(
                        x,
                        y,
                        marker=getMarkers(ActionList),
                        label=InfoString,
                        zorder=3,
                        markersize=20,
                    )
                    plt.annotate(
                        InfoString,
                        (x, y),
                        xytext=(0, 10),
                        textcoords="offset points",
                        rotation=55,
                        ha="left",
                        va="bottom",
                        fontsize=10,
                    )  # plot the legend
                # plt.legend(loc='lower right')
                lines_labels = {
                    line.get_label(): line for line in plt.gca().get_lines()
                }
                # plt.legend(lines_labels.values(), lines_labels.keys(), loc='lower right')

                # filter out the legend with _line0
                lines_labels = {
                    k: v for k, v in lines_labels.items() if not k.startswith("_line")
                }

                plt.subplots_adjust(right=0.7)
                plt.legend(
                    lines_labels.values(),
                    lines_labels.keys(),
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    labelspacing=2,
                )
            else:
                plt.plot(ScoreY, marker="o")

            # plt.margins(x=0.05, y=0.05)
            # set the axis to start from 0

            plt.xlabel("Timesteps")
            plt.ylabel("Graph Type (By Edges)")
            UniqueYPoints = list(np.unique(ScoreY))

            plt.yticks(UniqueYPoints, ListOfLabels)
            # show all xticks
            plt.xticks(np.arange(0, len(ScoreY), 1))

            if False:  # Whether to set x and y lims
                dx = (max(ScoreY) - min(ScoreY)) / 20
                dy = max(ScoreY) / 20

                plt.ylim(
                    -dy, self.si_graph_checker.DifferentGraphsByScore + (2 * dy)
                )  # Add to the end for the yticks to get a bit of space
                plt.xlim(-dx, len(ScoreY) + dx)

            plt.title(f"Social Influence Graphs Types over Ep (T{episode})")
            # plt.margins(y=0.05)
            # plot lines at self.CutOffPointsForGraphTypes, use self.colormap --- no longer any point because I do score now
            # for i in range(len(self.si_graph_checker.CutOffPointsForGraphTypes)):
            #     plt.axhline(y=self.si_graph_checker.CutOffPointsForGraphTypes[i]+0.5, color=self.si_graph_checker.colormap[i], linestyle='--')
            SavePath = (
                f"{self.gifs_dir}/SI_GraphType-Ep{episode}-Pruned{PercentilePrune}.png"
            )
            plt.tight_layout()
            plt.savefig(SavePath)
            # plt.show()
            plt.clf()
            plt.cla()

        ###### Histogram of Graph Types - score only
        hist = np.histogram(
            Y_GraphPerTimestep, bins=self.si_graph_checker.DifferentGraphsByScore
        )
        Whist = wandb.Histogram(np_histogram=hist)

        Y_GraphPerTimestep = np.array(Y_GraphPerTimestep)

        # Make an np.array of all known graph types
        ScoreGraph = np.zeros(self.si_graph_checker.DifferentGraphsByScore)
        # fill in from Y_GraphPerTimestep
        for score in Y_GraphPerTimestep:
            Index = self.si_graph_checker.MapScoreToIndex.get(score, -1)
            ScoreGraph[Index] += 1

        if self.CentralMemory.get(f"GraphTypeMemory_{PercentilePrune}", None) is None:
            self.CentralMemory[f"GraphTypeMemory_{PercentilePrune}"] = ScoreGraph
        else:
            # if the length of graphs increase, we need to add zero padding for graphs which did not exist before...
            MyPastGraph = self.CentralMemory[f"GraphTypeMemory_{PercentilePrune}"]
            if len(MyPastGraph) != len(ScoreGraph):
                # for now fully replace the counts
                self.CentralMemory[f"GraphTypeMemory_{PercentilePrune}"] = ScoreGraph
            else:
                self.CentralMemory[f"GraphTypeMemory_{PercentilePrune}"] += ScoreGraph

        ##### Heatmap of Graph Types and actions
        heatMapCounter = np.zeros(
            (len(self.actionTypes), self.si_graph_checker.DifferentGraphsByScore)
        )
        for actionSet, graphTypeScore in zip(actionsTaken, Y_GraphPerTimestep):
            for action in actionSet:
                actionIndex = self.action_to_index(action)
                TypeIndex = self.si_graph_checker.MapScoreToIndex.get(
                    graphTypeScore, -1
                )
                heatMapCounter[actionIndex, TypeIndex] += 1

        if PlotHeatMap:
            # matplotlib heatmap
            fig, ax = plt.subplots()
            im = ax.imshow(heatMapCounter, cmap="viridis")

            assert (
                len(self.si_graph_checker.statement)
                == self.si_graph_checker.DifferentGraphsByScore
            ), "The number of statements should be the same as the number of graph types"
            Labels = [
                i.replace(" edges", "E").replace("Type ", "")
                for i in self.si_graph_checker.statement
            ]  # add a blank for the last one
            XTicks = np.arange(
                0, self.si_graph_checker.DifferentGraphsByScore, 1
            )  # add a blank for the last one
            plt.xticks(XTicks, Labels, rotation=45, ha="right")
            plt.margins(x=0.05, y=0.05)

            plt.yticks(np.arange(len(self.actionTypes)), self.actionTypes)
            plt.xlabel("Graph Types")
            plt.ylabel("Team Maneuver")
            plt.title(f"SI Graph Type Team Maneuver Heatmap - T{episode}")
            # add the count numbers
            for i in range(self.si_graph_checker.DifferentGraphsByScore):
                for j in range(len(self.actionTypes)):
                    text = ax.text(
                        i,
                        j,
                        round(heatMapCounter[j, i]),
                        ha="center",
                        va="center",
                        color="white",
                    )

            HeatMapPath = f"{self.gifs_dir}/SI_GraphTypeJointActionHeatmap-Ep{episode}-Pruned{PercentilePrune}.png"
            plt.tight_layout()
            plt.savefig(HeatMapPath)
            # plt.show()
            plt.clf()
            plt.cla()

        else:
            HeatMapPath = None

        if self.CentralMemory.get(f"heatMapCounter_{PercentilePrune}", None) is None:
            self.CentralMemory[f"heatMapCounter_{PercentilePrune}"] = heatMapCounter
        else:
            MyPastGraph = self.CentralMemory[f"heatMapCounter_{PercentilePrune}"]
            # if len(MyPastGraph) != len(heatMapCounter):
            if MyPastGraph.shape != heatMapCounter.shape:
                self.CentralMemory[f"heatMapCounter_{PercentilePrune}"] = heatMapCounter
            else:
                self.CentralMemory[
                    f"heatMapCounter_{PercentilePrune}"
                ] += heatMapCounter  # Each time we add to it

        return SavePath, Whist, HeatMapPath

    def policy_analysis(self, policy, actions):
        """Takes for all timestep, return dictionary.
        Similar to smac_action_analysis."""
        if isinstance(policy, list) or isinstance(policy, torch.Tensor):
            policy = (
                convert_individual_to_joint(policy).cpu().numpy()
            )  # comes from prepped data
            policy = policy.squeeze()

        if len(policy.shape) == 4:  # Policy (18, 1, 3, 9)
            policy = policy.squeeze()  # come from game

        # if actions is a list # if it is a torch tensor
        if isinstance(actions, list) or isinstance(actions, torch.Tensor):
            actions = (
                convert_individual_to_joint(actions).cpu().numpy()
            )  # comes from prepped data
            actions = actions.squeeze()

        if len(actions.shape) == 4:  # Actions (18, 1, 3, 1)
            actions = actions.squeeze()  # come from game

        ListByTimestepActions = unstack(
            actions, 0
        )  # (batchsize, 3) where 3 is the number of agents, batchsize is the number of timesteps

        ## Plot the basic actions on a graph
        ListByTimestep = unstack(
            policy, 0
        )  # (batchsize, 3, policySize) where 3 is the number of agents, batchsize is the number of timesteps, policySize should be 9 for 3m

        Info = {}
        for i, current_policy_set in enumerate(ListByTimestep):
            Info[i] = {}
            for j, current_policy in enumerate(current_policy_set):
                # if not dead (no op), save policy with agentNumber
                action = ListByTimestepActions[i][j]
                alive = action != 0
                if alive:
                    Info[i][j] = (current_policy, action)  # save the policy
                else:
                    Info[i][j] = None  # save that the agent is dead

        return Info

    def plotPolicyIntoSimpleHistogram(
        self, ax, policy, color="b", plotXLabels=True, titles=False, plotActionName=True
    ):
        """small utility function to plot the policy into a histogram, matplotlib --- smaller compact version"""

        # if policy is len(9)
        # - 0 = No ops (agent is dead)
        # - 1 = Stop
        # - 2 = North
        # - 3 = South
        # - 4  = East
        # - 5 = West
        # - 6 = (agent 1 if they are in range)
        # - 7 = (agent 2 if they are in range)
        # - 8 = (agent 3 if they are in range)

        ## set y axis to be 0 to 1
        ax.set_ylim(0, 1)

        ax.bar(np.arange(len(policy)), policy, color=color, alpha=0.7)

        if plotXLabels:
            ax.set_xticks(np.arange(len(policy)))

            if len(policy) == len(self.ShortHandActions):
                ax.set_xticklabels(self.ShortHandActions)
            else:
                FullerActionSet = self.ShortHandActions + ["S5"]  # 5M policy display
                ax.set_xticklabels(FullerActionSet)

        else:
            ax.set_xticks([])

        if titles:
            ax.set_ylabel("Probability")
        else:
            ax.set_yticks([])

        if plotActionName:
            # Add big center label of argmax policy
            actionPred = np.argmax(policy)
            policy_argmax = self.actionTranslater(actionPred)
            ax.text(
                0.5,
                0.5,
                f"{policy_argmax}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
                color="black",
            )

        return ax

    def PlotSmallPlots(
        self,
        ax,
        policy,
        JS_Div,
        agent_predicting,
        counterfactual_action,
        lastRow=False,
        agent_predicted_for=None,
        current_time=None,
    ):
        """Dummy Data"""

        if JS_Div is not None:
            # thresholdColor = 'b' if JS_Div < 0.5 else 'r'

            if JS_Div <= 0.01:
                thresholdColor = "black"
            elif JS_Div < 0.3:
                thresholdColor = "blue"
            elif JS_Div < 0.6:
                thresholdColor = "orange"
            else:
                thresholdColor = "r"

            si_string = f"SI: {JS_Div:.2f}"
        else:
            thresholdColor = "gray"
            si_string = ""

        ax = self.plotPolicyIntoSimpleHistogram(
            ax, policy, color=thresholdColor, plotXLabels=lastRow, titles=False
        )
        counterfactual_action_name = self.actionTranslater(counterfactual_action)
        # agent_info = f"A{agent_predicted_for} policy in t{current_time+1} if A{agent_predicting} do {counterfactual_action_name} at t{current_time}"

        # agent_info = fr"$\tilde{{\pi}}_{{A{{{agent_predicting}}}, ({{{counterfactual_action_name}}}), T{{{current_time}}}}}^{{A{{{agent_predicted_for}}}, T{{{current_time+1}}}}}$"
        agent_info = rf"$\tilde{{\pi}}^{{{current_time+1}}}_{{{agent_predicted_for}}} ({{{counterfactual_action_name}}}^{{{current_time}}}_{{{agent_predicting}}})$"

        if si_string != "":
            # Add text to the left
            ax.text(
                0.05,
                0.9,
                si_string,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=8,
                color="black",
                weight="bold",
            )

        # Add text to the right

        ax.text(
            0.95,
            0.88,
            agent_info,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=12,
            color="black",
            usetex=True,
        )

        return ax

    def actionTranslater(self, action):
        # make sure action is int
        action = int(action)

        if not isinstance(action, int) and action.size == 1:
            action = action.item()
            # make float to int
            action = int(action)

        if action < len(self.ShortHandActions):
            return self.ShortHandActions[action]
        else:
            FullerActionSet = self.ShortHandActions + ["S5"]  # 5M policy display
            # print("Action: ", action, "FullerActionSet: ", FullerActionSet)
            return FullerActionSet[action]

    def plotPolicyIntoHistogram(
        self,
        ax,
        policy,
        action,
        agent_id,
        plotXLabelsLong=True,
        titles=True,
        plotActionName=True,
        hideTicks=False,
    ):
        """small utility function to plot the policy into a histogram, matplotlib"""

        # if policy is len(9)
        # - 0 = No ops (agent is dead)
        # - 1 = Stop
        # - 2 = North
        # - 3 = South
        # - 4  = East
        # - 5 = West
        # - 6 = (agent 1 if they are in range)
        # - 7 = (agent 2 if they are in range)
        # - 8 = (agent 3 if they are in range)

        # print("Policy: ", policy)
        # print("Action: ", action)
        # can be numpy int
        # If `action` is supposed to contain a single integer value
        if action.size == 1:
            action = action.item()
            # make float to int
            action = int(action)
        assert np.issubdtype(type(action), np.integer) or isinstance(
            action, int
        ), f"Action should be an integer but it is {type(action)}, {action}"
        ## set y axis to be 0 to 1
        ax.set_ylim(0, 1)

        ax.bar(np.arange(len(policy)), policy, color="b", alpha=0.7)
        # Action is the action that was taken, make it red
        ax.bar(action, policy[action], color="r", alpha=0.7)

        if plotXLabelsLong:
            ax.set_xticks(np.arange(len(policy)))
            ActionsRemaining = len(policy) - 6  # Should all be shoot enemy actions
            ListOfActionDefinitions = [
                "No Ops",
                "Stop",
                "North",
                "South",
                "East",
                "West",
            ] + [f"Shoot E{i+1}" for i in range(ActionsRemaining)]
            ax.set_xticklabels(ListOfActionDefinitions, rotation=45)
        else:
            ax.set_xticks(np.arange(len(policy)))
            # ax.set_xticklabels(self.ShortHandActions)
            if len(policy) == len(self.ShortHandActions):
                ax.set_xticklabels(self.ShortHandActions)
            else:
                FullerActionSet = self.ShortHandActions + ["S5"]  # 5M policy display
                ax.set_xticklabels(FullerActionSet)

        if titles:
            ax.set_title(f"Agent {agent_id} Policy")
            ax.set_ylabel("Probability")

        if plotActionName:
            # Add big center label of argmax policy
            actionPred = np.argmax(policy)
            policy_argmax = self.actionTranslater(actionPred)
            ax.text(
                0.5,
                0.5,
                f"{policy_argmax}",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
                color="black",
            )

        if hideTicks:
            ax.set_xticks([])
            ax.set_yticks([])

        return ax

    def plot_central_memory(self, episode=0, percentileToPruneList=[30]):
        for PercentilePrune in percentileToPruneList:
            fig, ax = plt.subplots()
            im = ax.imshow(
                self.CentralMemory[f"heatMapCounter_{PercentilePrune}"], cmap="viridis"
            )
            # plt.xticks(np.arange(self.si_graph_checker.DifferentGraphsByScore), [i.replace(" edges", "E") for i in self.si_graph_checker.statement], rotation=45, ha='right')
            Labels = [
                i.replace(" edges", "E").replace("Type ", "")
                for i in self.si_graph_checker.statement
            ]
            XTicks = np.arange(0, self.si_graph_checker.DifferentGraphsByScore, 1)
            plt.xticks(XTicks, Labels, rotation=45, ha="right")
            plt.yticks(np.arange(len(self.actionTypes)), self.actionTypes)
            plt.xlabel("Graph Types")
            plt.ylabel("Team Maneuver")
            # add the count numbers
            for i in range(self.si_graph_checker.DifferentGraphsByScore):
                for j in range(len(self.actionTypes)):
                    text = ax.text(
                        i,
                        j,
                        round(
                            self.CentralMemory[f"heatMapCounter_{PercentilePrune}"][
                                j, i
                            ]
                        ),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=6,
                    )

            HeatMapPath = f"{self.gifs_dir}/SI_GraphTypeJointActionHeatmap-Ep{episode}_{PercentilePrune}.png"
            plt.tight_layout()
            plt.savefig(HeatMapPath)
            # plt.show()
            plt.clf()
            plt.cla()

        ### GraphTypeMemory
        histogram = np.histogram(
            self.CentralMemory[f"GraphTypeMemory_{PercentilePrune}"],
            bins=self.si_graph_checker.DifferentGraphsByScore,
            density=True,
        )
        Whist = wandb.Histogram(np_histogram=histogram)

        ### All Graphs
        si_diagrams = (
            self.si_graph_checker.produce_big_plot()
        )  # Not currently saved to wandb

        return HeatMapPath, Whist, si_diagrams

    def save_si(self, save_dir, episode=0, episodeLabel=False):
        """Save policy's actor and critic networks."""
        if self.run_mixed_population:
            self.social_influence_net_A.save(save_dir, episode)

            if episodeLabel:
                self.social_influence_net_A.save(save_dir, episode, withEpName=True)

            self.social_influence_net_B.save(save_dir, episode)

            if episodeLabel:
                self.social_influence_net_B.save(save_dir, episode, withEpName=True)
        else:
            # TODO no saving for fixed and learning? because we don't care since loading for eval?
            self.social_influence_net.save(save_dir, episode)

            if episodeLabel:
                self.social_influence_net.save(save_dir, episode, withEpName=True)

    def train(
        self,
        joint_obs_A,
        actions_A,
        true_policies_A,
        joint_obs_B=None,
        actions_B=None,
        true_policies_B=None,
        AGENTS_CAN_DIE=False,
    ):
        if self.run_mixed_population:
            joint_obs_A = joint_obs_A
            actions_A = actions_A.clone()
            true_policies_A = true_policies_A

            
            

            

            if true_policies_B is None:
                print("WARNING SI manager not getting required input for teamB")
            joint_obs_B = joint_obs_B.clone()
          
            actions_B = actions_B.clone()
            
            TrainInfoA = self.social_influence_net_A.train_self(
                joint_obs_A, actions_A, true_policies_A, AGENTS_CAN_DIE=AGENTS_CAN_DIE
            )
            TrainInfoB = self.social_influence_net_B.train_self(
                joint_obs_B, actions_B, true_policies_B, AGENTS_CAN_DIE=AGENTS_CAN_DIE
            )
            # append an indicator of which network was used
            TrainInfoA = {k + "_(Team A)": v for k, v in TrainInfoA.items()}
            TrainInfoB = {k + "_(Team B)": v for k, v in TrainInfoB.items()}
            return {**TrainInfoA, **TrainInfoB}

        else:
            return self.social_influence_net.train_self(
                joint_obs_A, actions_A, true_policies_A, AGENTS_CAN_DIE=AGENTS_CAN_DIE
            )


class GroupSocialInfluence(nn.ModuleList):
    ## Holds a group of social influence networks
    def __init__(self, params):
        super().__init__()
        self.gifs_dir = params["gifs_dir"]
        self.num_env_steps = params["num_env_steps"]
        self.num_agents = params["num_agents"]
        # self.optimizers_si_list = []
        # self.scheduler_si_list = []
        self.social_influence_networks = []
        assert self.num_agents > 0, "Number of agents should be greater than 0"
        # is an integer
        assert self.num_agents == int(
            self.num_agents
        ), "Number of agents should be an integer"
        for i in range(self.num_agents):
            otherAgentPolicyDims = [
                params["action_space"] for _ in range(self.num_agents - 1)
            ]

            self.social_influence_networks.append(
                SocialInfluencePredictor(
                    otherAgentPolicyDims=otherAgentPolicyDims,
                    n_obs=params["n_obs"],
                    action_dim=params["action_dim"],
                    n_hidden_units=params["n_hidden_units"],
                    num_agents=params["num_agents"],
                    discrete_actions=params["discrete_actions"],
                    social_influence_n_counterfactuals=params[
                        "social_influence_n_counterfactuals"
                    ],
                    si_loss_type=params["si_loss_type"],
                    only_use_argmax_policy=params["only_use_argmax_policy"],
                )
            )

        self.si_loss_type = params["si_loss_type"]
        self.only_use_argmax_policy = params["only_use_argmax_policy"]

        # self.TwoDifferentTeamEnvs_TeamBAdjustments = params.get("TwoDifferentTeamEnvs_TeamBAdjustments", False)

        self.config = params

        if self.si_loss_type == "kl" or self.si_loss_type == "bce":
            learning_rate_si = 3e-4
            weight_decay_si = 1e-5
            optimizer_epsilon_si = 1e-5  # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        elif self.si_loss_type == "js":
            learning_rate_si = 5e-3
            weight_decay_si = 1e-4
            optimizer_epsilon_si = 1e-4  # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

        optimizers_si_list = []
        scheduler_si_list = []
        for social_influence_network in self.social_influence_networks:
            optimizer_si = optim.Adam(
                social_influence_network.parameters(),
                lr=learning_rate_si,
                weight_decay=weight_decay_si,
                eps=optimizer_epsilon_si,
            )
            scheduler_si = torch.optim.lr_scheduler.LambdaLR(
                optimizer_si, lr_lambda=lambda step: 1 - step / self.num_env_steps
            )
            optimizers_si_list.append(optimizer_si)
            scheduler_si_list.append(scheduler_si)

        self.optimizers_list = optimizers_si_list
        self.schedulers_list = scheduler_si_list

    def save(self, save_dir, timestep=0, withEpName=False):
        new_save_dir = os.path.join(save_dir, "social_influence_main")
        os.makedirs(new_save_dir, exist_ok=True)
        for idx, social_influence_network in enumerate(self.social_influence_networks):
            torch.save(
                social_influence_network.state_dict(),
                str(new_save_dir) + f"/social_influence_network_{str(idx)}.pt",
            )

        if withEpName:
            NumSteps = f"{timestep/1000000:.1f}M"
            # make folder with episode name
            new_save_dir = os.path.join(save_dir, f"social_influence_ts{str(NumSteps)}")
            # if folder does not exist make it
            os.makedirs(new_save_dir, exist_ok=True)
            for idx, social_influence_network in enumerate(
                self.social_influence_networks
            ):
                torch.save(
                    social_influence_network.state_dict(),
                    str(new_save_dir) + f"/social_influence_network_{str(idx)}.pt",
                )

    def load(self, model_dir):
        for idx, social_influence_network in enumerate(self.social_influence_networks):
            policy_state_dict = torch.load(
                str(model_dir) + f"/social_influence_network_{str(idx)}.pt"
            )
            social_influence_network.load_state_dict(policy_state_dict)

    # def adjustInCaseDifferentEnv(self, inputX, Xtype="obs"):
    #     out = inputX
    #     if self.TwoDifferentTeamEnvs_TeamBAdjustments:
    #         if Xtype == "obs":
    #             # pad observations → into network
    #         elif Xtype == "obs":
    #             # pad invalid actions → into network

    #             # cut size of actions → out of system

    #             # pad policies → into network

    #     return out

    def get_loss(
        self, IndividualObs, IndividualActions, true_policies, AGENTS_CAN_DIE=False
    ):
        """Assume that the correction for timestep is already done, and so is the reshaping of the data"""
        train_info = {}

        # Make this list of joint obs into individual obs list, e.g. switch from [timestep, agent, obs] to [agent, timestep, obs]
        # IndividualObs = self.convert_joint_to_individual(joint_obs)
        # JointActions = self.convert_joint_to_individual(actions)
        # IndividualActions = JointActions.float()

        ### Get the original correct policies we are trying to predict that does use the skill
        # with torch.no_grad():
        # # Make into Policy Observations
        # if SKILLS_USED:
        #     skills = skills
        #     skillInBinary = binary_encode(skills, self.config["size_of_binary_encoding"])

        #     # Convert to tensor
        #     skillInBinary = torch.tensor(skillInBinary, dtype=torch.float32).squeeze()
        if self.only_use_argmax_policy:
            true_policies = softmax_to_onehot(true_policies)

        social_influence_loss_total = 0
        for agent_id in range(self.num_agents):
            obs = IndividualObs[agent_id]
            indiv_actions = IndividualActions[agent_id]

            network = self.social_influence_networks[agent_id]

            social_influence_loss = network.calc_social_influence_loss(
                obs, indiv_actions, true_policies, agent_id, AGENTS_CAN_DIE
            )

            social_influence_loss_total += social_influence_loss.item()

        social_influence_loss_total /= self.num_agents

        # train_info = {"social_influence_loss": social_influence_loss_total}

        return social_influence_loss_total

    def train_self(
        self, IndividualObs, IndividualActions, true_policies, AGENTS_CAN_DIE=False
    ):
        """Assume that the correction for timestep is already done, and so is the reshaping of the data"""
        train_info = {}

        assert (
            IndividualObs.shape[1] == IndividualActions.shape[1]
        ), "train_self Individual obs and actions should have the same batch size"
        assert (
            IndividualObs.shape[1] == true_policies.shape[0]
        ), "train_self Individual obs and policies should have the same batch size"
        assert (
            IndividualObs.shape[0] == self.num_agents
        ), "train_self Individual obs should have agent number as the first dimension"
        assert (
            IndividualActions.shape[0] == self.num_agents
        ), "train_self Individual actions should have agent number as the first dimension"
        assert (
            true_policies.shape[1] == self.num_agents
        ), "train_self true_policies should have agent number as the first dimension"

        if self.only_use_argmax_policy:
            true_policies = softmax_to_onehot(true_policies)

        ### Get the original correct policies we are trying to predict that does use the skill
        # with torch.no_grad():
        #     # Make into Policy Observations
        #     if SKILLS_USED:
        #         skills = skills
        #         skillInBinary = binary_encode(skills, self.config["size_of_binary_encoding"])

        #         # Convert to tensor
        #         skillInBinary = torch.tensor(skillInBinary, dtype=torch.float32).squeeze()

        # assert num_agents == self.num_agents, "Number of agents in buffer should be the same as the number of agents in the social influence network"
        social_influence_loss_total = 0
        social_influence_grad_norm_total = 0
        for agent_id in range(self.num_agents):
            obs = IndividualObs[agent_id]
            indiv_actions = IndividualActions[agent_id]

            network = self.social_influence_networks[agent_id]
            optimizer = self.optimizers_list[agent_id]
            scheduler = self.schedulers_list[agent_id]

            social_influence_loss = network.calc_social_influence_loss(
                obs, indiv_actions, true_policies, agent_id, AGENTS_CAN_DIE
            )

            optimizer.zero_grad()
            social_influence_loss.backward()
            social_influence_grad_norm = torch.nn.utils.clip_grad_norm_(
                network.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            # detect there is no inf social influence loss
            if (
                torch.isnan(social_influence_loss).any()
                or torch.isnan(social_influence_grad_norm).any()
            ):
                print(
                    f"WARNING Nan in social_influence_loss {torch.isnan(social_influence_loss).any()} or social_influence_grad_norm {torch.isnan(social_influence_grad_norm).any()}"
                )
                # raise error
                continue

            if (
                torch.isinf(social_influence_loss).any()
                or torch.isinf(social_influence_grad_norm).any()
            ):
                print(
                    f"WARNING Inf in social_influence_loss {torch.isinf(social_influence_loss).any()} or social_influence_grad_norm {torch.isinf(social_influence_grad_norm).any()}"
                )
                # raise error
                continue

            # print("social_influence_loss", social_influence_loss)
            social_influence_loss_total += social_influence_loss.item()
            social_influence_grad_norm_total += social_influence_grad_norm.item()

        social_influence_loss_total /= self.num_agents
        social_influence_grad_norm_total /= self.num_agents

        # print("social_influence_loss_total", social_influence_loss_total)

        train_info = {
            "social_influence_loss": social_influence_loss_total,
            "social_influence_grad_norm": social_influence_grad_norm_total,
        }

        return train_info

    def calc_social_influence_reward_with_counterfactuals_returned(
        self, obs, actions, weightOfPenalty=0.5
    ):
        """
        Calculate the social influence reward for a group of agents.

        Parameters:
        obs (np.array): Observations from the environment.
        actions (np.array): Actions taken by the agents.
        shape (tuple): Shape of the observations.
        weightOfPenalty (float, optional): Weight of the penalty for the reward calculation. Defaults to 0.5.

        Returns:
        SocialInfluenceAgentReward (np.array): Array of social influence rewards for each agent.
        BiDirectionalGraph (np.array): Bi-directional graph representing the social influence between agents.
        """

        assert (
            obs.shape[1] == actions.shape[1]
        ), "calc_social_influence_reward_group Individual obs and actions should have the same batch size"
        assert (
            obs.shape[0] == self.num_agents
        ), "calc_social_influence_reward_group Individual obs should have agent number as the first dimension"
        assert (
            actions.shape[0] == self.num_agents
        ), "calc_social_influence_reward_group Individual actions should have agent number as the first dimension"

        reformat_actions = convert_individual_to_joint(actions)
        aliveMask = np.ones_like(reformat_actions)
        # if action = 0 then agent is dead
        # aliveMask[reformat_actions == 0] = 0 #TODO All agents are alive throug the game

        all_agent_social_influence = []
        counterfactual_policies = []
        pred_true_policies = []
        for count in range(self.num_agents):
            individualObs = obs[count]
            individualActions = actions[count]
            social_influence_metric, counterfactual_policy, pred_true_policy = (
                self.calc_social_influence_metric_with_counterfactuals_singleAgentPolicy(
                    individualObs, individualActions, count, AGENT_CAN_DIE=True
                )
            )  # this is batchsize, other agent number, counterfactuals
            all_agent_social_influence.append(social_influence_metric)
            counterfactual_policies.append(counterfactual_policy)
            pred_true_policies.append(pred_true_policy)

        return (all_agent_social_influence, counterfactual_policies, pred_true_policies)

    def calc_social_influence_reward_group(self, obs, actions, weightOfPenalty=0.5):
        """
        Calculate the social influence reward for a group of agents.

        Parameters:
        obs (np.array): Observations from the environment.
        actions (np.array): Actions taken by the agents.
        shape (tuple): Shape of the observations.
        weightOfPenalty (float, optional): Weight of the penalty for the reward calculation. Defaults to 0.5.

        Returns:
        SocialInfluenceAgentReward (np.array): Array of social influence rewards for each agent.
        BiDirectionalGraph (np.array): Bi-directional graph representing the social influence between agents.
        """
        # print(obs.shape)
        # print(actions.shape)
        assert (
            obs.shape[1] == actions.shape[1]
        ), "calc_social_influence_reward_group Individual obs and actions should have the same batch size"
        assert (
            obs.shape[0] == self.num_agents
        ), "calc_social_influence_reward_group Individual obs should have agent number as the first dimension"
        assert (
            actions.shape[0] == self.num_agents
        ), "calc_social_influence_reward_group Individual actions should have agent number as the first dimension"

        reformat_actions = convert_individual_to_joint(actions)
        aliveMask = np.ones_like(reformat_actions)
        # if action = 0 then agent is dead
        # aliveMask[reformat_actions == 0] = 0  #TODO All agents are alive through the game

        # print("shape of obs", obs.shape)
        # print("shape of actions", actions.shape)

        all_agent_social_influence = []
        variance = []
        for count in range(self.num_agents):
            individualObs = obs[count]
            individualActions = actions[count]
            social_influence_reward, social_influence_variation = (
                self.calc_social_influence_reward_singleAgentPolicy(
                    individualObs, individualActions, count, AGENT_CAN_DIE=True
                )
            )  # this is batchsize and agent number
            all_agent_social_influence.append(social_influence_reward)
            variance.append(social_influence_variation)

        ##### pairwise social influence - sequential except remove yourself comparison
        metrics = []
        for keyMetric in [all_agent_social_influence, variance]:
            byAgent = [a.permute(1, 0) for a in keyMetric]  # other agents, batchsize
            ## make graph
            batchsize = keyMetric[0].shape[0]

            BiDirectionalGraph = np.zeros(
                (self.num_agents, self.num_agents, batchsize)
            )  # To make it easier to access

            # Fill up the graph
            for my_id in range(self.num_agents):
                for other_agent in range(self.num_agents):
                    if my_id == other_agent:
                        continue

                    ## Note these ids are just for secondary list predicting the other agent
                    # if the other agent is before me, -1 to my id for querying them
                    # if they are after me, -1 to their id for querying them
                    MyIndexID = my_id
                    OtherAgentID = other_agent

                    myActions = actions[my_id]
                    OtherActions = actions[other_agent]

                    # Assign meDead where myActions equals 0
                    meDead = (myActions == 0).nonzero(as_tuple=True)
                    otherAgentDead = (OtherActions == 0).nonzero(as_tuple=True)

                    if other_agent < my_id:
                        MyIndexID = my_id - 1
                    elif other_agent > my_id:
                        OtherAgentID = other_agent - 1

                    myPredictions = byAgent[my_id][OtherAgentID]  # agent, batchsize
                    otherPredictions = byAgent[other_agent][
                        MyIndexID
                    ]  # agent, batchsize

                    # Set myPredictions to 0 where meDead is True
                    myPredictions.index_fill_(0, meDead[0], 0)
                    # Set otherPredictions to 0 where otherAgentDead is True
                    otherPredictions.index_fill_(0, otherAgentDead[0], 0)

                    BiDirectionalGraph[my_id, other_agent, :] = myPredictions
                    BiDirectionalGraph[other_agent, my_id, :] = otherPredictions

            # remove influence prediction if other agent if dead
            for t in range(batchsize):
                for i in range(self.num_agents):
                    if not aliveMask[t, i, 0]:  # If the agent is dead
                        # Set SI from the agent to all others to zero
                        BiDirectionalGraph[i, :, t] = 0
                        # Set SI to the agent from all others to zero
                        BiDirectionalGraph[:, i, t] = 0

            # Calculate the reward, per agent, per timestep

            SocialInfluenceAgentReward = np.zeros((self.num_agents, batchsize))
            for my_id in range(self.num_agents):
                MyReward = np.zeros((batchsize))
                AliveAgents = np.zeros_like(MyReward)
                for other_agent in range(self.num_agents):
                    if my_id != other_agent:
                        myPredictionsAboutOther = BiDirectionalGraph[
                            my_id, other_agent, :
                        ]
                        otherPrediction = BiDirectionalGraph[other_agent, my_id, :]
                        rwd = (myPredictionsAboutOther + otherPrediction) - (
                            weightOfPenalty
                            * np.abs(myPredictionsAboutOther - otherPrediction)
                        )
                        MyReward += rwd

                        # Calculate AliveAgents as the sum of where myPredictionsAboutOther and otherPrediction are both non-zero
                        AliveAgentsCount = np.sum(
                            (myPredictionsAboutOther != 0) & (otherPrediction != 0)
                        )
                        AliveAgents += AliveAgentsCount
                # Normalize
                MyReward = MyReward / AliveAgents
                SocialInfluenceAgentReward[my_id, :] = MyReward

            metrics.append((SocialInfluenceAgentReward, BiDirectionalGraph))

        return metrics

    def calc_social_influence_reward_singleAgentPolicy(
        self, individualObs, individualTrueAction, agent_id, AGENT_CAN_DIE=False
    ):
        assert (
            individualObs.shape[0] == individualTrueAction.shape[0]
        ), "calc_social_influence_reward_singleAgentPolicy Individual obs and actions should have the same batch size"
        # TODO make this work for multiple agents with different policy networks
        # No need gradients because this reward is generated by social influence network for the DIAYN network
        with torch.no_grad():
            ### Social influence stuff here - if current actions were different, how would next step change?
            # For each agent, compute alternative other agent policy based on every counterfactual possible

            ######################## AGENT, -----  TIMESTEP (BATCHSIZE)

            if AGENT_CAN_DIE:
                # Get the indices where the action is not 0
                non_noop_indices = np.where(individualTrueAction != 0)[0]

                # num_non_noop_indices = len(non_noop_indices)
                # print("Number of non-noop indices:", num_non_noop_indices)

                # Select the corresponding elements from the action, observation, and ListOfTruePoliciesOfAgents arrays
                processedTrueActionsBatched = individualTrueAction[
                    non_noop_indices
                ].clone()
                processedObsBatched = individualObs[non_noop_indices].clone()
            else:
                processedTrueActionsBatched = individualTrueAction
                processedObsBatched = individualObs
            # print("processedTrueActionsBatched", processedTrueActionsBatched.shape)
            # print("processedObsBatched", processedObsBatched.shape)
            predictedPoliciesForTrueAction = self.social_influence_networks[agent_id](
                x=processedObsBatched,
                action=processedTrueActionsBatched,  # true action at timestep do a whole batch
            )  #  predicted with true action input policies

            ##### Unbind by zero to iterate over the batch
            if self.config["discrete_actions"]:
                # softmax_tensors = predictedPoliciesForTrueAction[2]
                softmax_tensors = predictedPoliciesForTrueAction[
                    2
                ]  # torch.unbind(, dim=1)[0]

                # batch size, other agent num, policy size

                batch_predicted_true_dist_i = [
                    softmax_tensors
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor

                batch_predicted_true_dist_i = torch.stack(
                    batch_predicted_true_dist_i, dim=0
                )  # counterfactuals, batch size, other agent num, policy size

                # print("softmax_tensors", softmax_tensors.shape)
                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape)

                batch_predicted_true_dist_i = batch_predicted_true_dist_i.permute(
                    1, 0, 2, 3
                )

                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape) # batch size, counterfactuals, other agent num, policy size

                MyTrueActions = processedTrueActionsBatched.squeeze(1)
                # allPossibleActions = torch.arange(1, self.config["action_space"])
                allPossibleActions = torch.arange(
                    1, self.config["social_influence_n_counterfactuals"] + 2
                )  # because we remove the noop action, true action will be removed below, but in case of team B, we don't want the last shooting action, so just use the same social_influence_n_counterfactuals set by team A (adding 2 for the no op that arange removes and the true action removed below)
                BatchCounterfactualActions = []
                for true_action in MyTrueActions:
                    true_action = (
                        true_action.long()
                    )  # Ensure true_action is of type Long
                    counterfactual_actions = allPossibleActions[
                        allPossibleActions != true_action
                    ]
                    BatchCounterfactualActions.append(counterfactual_actions)

                # no need to permute to get batch size, counterfactual, action dim
                BatchCounterfactualActionsForThisBatchOfAction = torch.concat(
                    BatchCounterfactualActions, dim=0
                ).float()
                # unsqueeze last dimension
                BatchCounterfactualActionsForThisBatchOfAction = (
                    BatchCounterfactualActionsForThisBatchOfAction.unsqueeze(-1)
                )
                # print("BatchCounterfactualActionsForThisBatchOfAction", BatchCounterfactualActionsForThisBatchOfAction.shape)

            else:
                predicted_true_mu_i = torch.unbind(
                    predictedPoliciesForTrueAction[0], dim=1
                )[0]
                predicted_true_std_i = torch.unbind(
                    predictedPoliciesForTrueAction[1], dim=1
                )[
                    0
                ]  # get per agent since this is a tuple

                batch_predicted_true_mu_i = [
                    predicted_true_mu_i
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor
                batch_predicted_true_mu_i = torch.stack(
                    batch_predicted_true_mu_i, dim=0
                ).permute(1, 0, 2)
                batch_predicted_true_std_i = [
                    predicted_true_std_i
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor
                batch_predicted_true_std_i = torch.stack(
                    batch_predicted_true_std_i, dim=0
                ).permute(1, 0, 2)

                batch_predicted_true_dist_i = Normal(
                    batch_predicted_true_mu_i, batch_predicted_true_std_i
                )

                # unbind along the batch dimension
                # [bs, count_num, action_dim]
                BatchCounterfactualActionsForThisBatchOfAction = [
                    torch.randn_like(processedTrueActionsBatched)
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]
                BatchCounterfactualActionsForThisBatchOfAction = torch.concat(
                    BatchCounterfactualActionsForThisBatchOfAction, dim=0
                ).float()  # [bs*num,action_dim]
            # [bs, pov,pov,channel]
            # print(processedObsBatched.shape)
            Batch_true_obs_batchList = [
                processedObsBatched
                for i in range(self.config["social_influence_n_counterfactuals"])
            ]
            Batch_true_obs_batchList = torch.concat(
                Batch_true_obs_batchList, dim=0
            )  # TODO Ask Pamela once

            predictedPoliciesForCounterFactualAction = self.social_influence_networks[
                agent_id
            ](
                x=Batch_true_obs_batchList,  # batch size, counterfactuals, pov,pov,n_channels
                action=BatchCounterfactualActionsForThisBatchOfAction,  # this is a single action with batchsize 1 # batch size, counterfactuals, action size
            )
            # print(predictedPoliciesForCounterFactualAction[2].shape)
            # batch size, other agents, counterfactual size, policy size
            ## JS divergence
            if self.config["discrete_actions"]:
                predicted_counterfactual_dist = torch.reshape(
                    predictedPoliciesForCounterFactualAction[2],
                    (
                        -1,
                        self.config["social_influence_n_counterfactuals"],
                        self.num_agents - 1,
                        predictedPoliciesForCounterFactualAction[2].shape[-1],
                    ),
                )

                # predicted_counterfactual_dist = torch.unbind(predictedPoliciesForCounterFactualAction[2], dim=1)[0]
                # predicted_counterfactual_dist = predictedPoliciesForCounterFactualAction[2].permute(0, 2, 1, 3) #TODO
                # print("predicted_counterfactual_dist", predicted_counterfactual_dist.shape) # batch size, counterfactual size, other agents, policy size
                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape) # batch size, counterfactual size, other agents, policy size
                # assert that valid probability distributions (i.e., they are non-negative and sum to 1)
                assert (
                    predicted_counterfactual_dist.shape
                    == batch_predicted_true_dist_i.shape
                ), "Predicted counterfactual dist and batch predicted true dist should have the same shape"
                # assert (predicted_counterfactual_dist >= 0).all(), "Predicted counterfactual dist should be non-negative"
                # assert (torch.abs(predicted_counterfactual_dist.sum(dim=-1) -1 ) < 1e-6).all(), f"Predicted counterfactual dist should sum to 1, actual sum is {predicted_counterfactual_dist.sum(dim=-1)}"
                # assert (batch_predicted_true_dist_i >= 0).all(), "Batch predicted true dist should be non-negative"
                # assert (torch.abs(batch_predicted_true_dist_i.sum(dim=-1) -1 ) < 1e-6).all(), f"Batch predicted true dist should sum to 1, actual sum is {batch_predicted_true_dist_i.sum(dim=-1)}"
                LOGITS = False
                if LOGITS:
                    #### This is for KL divergence
                    predNormDist = Categorical(
                        logits=predicted_counterfactual_dist
                    )  # this one should be logprobs because we forward with logits
                    trueNormDist = Categorical(
                        logits=batch_predicted_true_dist_i
                    )  # This one should be logprobs because we prep data above

                    # Convert logits to probabilities
                    # true_probs = F.softmax(batch_predicted_true_dist_i, dim=-1)
                    # pred_probs = F.softmax(predicted_counterfactual_dist, dim=-1)
                    true_probs = torch.exp(batch_predicted_true_dist_i)
                    pred_probs = torch.exp(predicted_counterfactual_dist)
                else:

                    predNormDist = Categorical(
                        probs=predicted_counterfactual_dist
                    )  # this one should be logprobs because we forward with logits
                    trueNormDist = Categorical(
                        probs=batch_predicted_true_dist_i
                    )  # This one should be logprobs because we prep data above

                    true_probs = batch_predicted_true_dist_i
                    pred_probs = predicted_counterfactual_dist

                mixture_dist = 0.5 * (true_probs + pred_probs)
                # mixture_dist = mixture_dist / mixture_dist.sum(dim=-1, keepdim=True) # normalize
                # mixture_dist_logits = F.log_softmax(mixture_dist, dim=-1)
                # mixtureDist = Categorical(logits=mixture_dist_logits)
                mixtureDist = Categorical(probs=mixture_dist)

                js_divergence = 0.5 * (
                    kl.kl_divergence(trueNormDist, mixtureDist)
                    + kl.kl_divergence(predNormDist, mixtureDist)
                )

                # js divergence is non-negative so clamp
                js_divergence = torch.clamp(js_divergence, min=0)

                # singleLoss = js_divergence.mean()

                # js_divergence = 0.5 * ((kl.kl_divergence(Categorical(probs=batch_predicted_true_dist_i), Categorical(probs=mixture_dist)) + kl.kl_divergence(Categorical(probs=predicted_counterfactual_dist), Categorical(probs=mixture_dist))))

                # TODO put this back
                # assert (js_divergence >= 0).all(), f"JS divergence should be non-negative but it is {js_divergence}"
            else:
                predicted_counterfactual_mu_i = torch.unbind(
                    predictedPoliciesForCounterFactualAction[0], dim=1
                )[0]
                predicted_counterfactual_std_i = torch.unbind(
                    predictedPoliciesForCounterFactualAction[1], dim=1
                )[0]
                predicted_counterfactual_dist = Normal(
                    predicted_counterfactual_mu_i, predicted_counterfactual_std_i
                )
                kl_divergence = kl.kl_divergence(
                    batch_predicted_true_dist_i, predicted_counterfactual_dist
                )
                print("TODO change this to js divergence for continuous actions")
                # TODO contd broken for now
                assert (
                    kl_divergence >= 0
                ), "KL divergence should be non-negative but it is {kl_divergence}"

            if self.config["discrete_actions"]:
                social_influence_reward = js_divergence.mean(dim=1)

                social_influence_variance = js_divergence.var(dim=1)

                # JS divergence should be non-negative so if negative is precision error
            else:
                social_influence_reward = kl_divergence.mean(dim=1).mean(dim=1)

                social_influence_variance = js_divergence.var(dim=1)

            # assert len(social_influence_reward) == processedObsBatched.shape[0]

            # Assign the first dimension of individualTrueAction to batchsize
            batchsize = individualTrueAction.shape[
                0
            ]  # Calculate the number of zeroes needed to pad social_influence_reward to the desired length
            padding_length = batchsize - social_influence_reward.size(0)

            # Create a tensor of zeroes with the required size
            zero_padding = torch.zeros(
                (padding_length,) + social_influence_reward.shape[1:]
            )

            # Concatenate social_influence_reward and the tensor of zeroes along dimension 0
            social_influence_reward_padded = torch.cat(
                (social_influence_reward, zero_padding), dim=0
            )

            social_influence_variance_padded = torch.cat(
                (social_influence_variance, zero_padding), dim=0
            )

            # assert there are no negatives in the reward
            # assert (social_influence_reward_padded >= 0).all(), "Social influence reward should be non-negative, {social_influence_reward_padded}"

            return (
                social_influence_reward_padded,
                social_influence_variance_padded,
            )  # this should be a list per timestep

    def calc_social_influence_metric_with_counterfactuals_singleAgentPolicy(
        self, individualObs, individualTrueAction, agent_id, AGENT_CAN_DIE=False
    ):
        assert (
            individualObs.shape[0] == individualTrueAction.shape[0]
        ), "calc_social_influence_reward_singleAgentPolicy Individual obs and actions should have the same batch size"
        # TODO make this work for multiple agents with different policy networks
        # No need gradients because this reward is generated by social influence network for the DIAYN network
        with torch.no_grad():
            ### Social influence stuff here - if current actions were different, how would next step change?
            # For each agent, compute alternative other agent policy based on every counterfactual possible

            ######################## AGENT, -----  TIMESTEP (BATCHSIZE)

            if AGENT_CAN_DIE:
                # Get the indices where the action is not 0
                non_noop_indices = np.where(individualTrueAction != 0)[0]

                # num_non_noop_indices = len(non_noop_indices)
                # print("Number of non-noop indices:", num_non_noop_indices)

                # Select the corresponding elements from the action, observation, and ListOfTruePoliciesOfAgents arrays
                processedTrueActionsBatched = individualTrueAction[
                    non_noop_indices
                ].clone()
                processedObsBatched = individualObs[non_noop_indices].clone()
            else:
                processedTrueActionsBatched = individualTrueAction
                processedObsBatched = individualObs

            predictedPoliciesForTrueAction = self.social_influence_networks[agent_id](
                x=processedObsBatched,
                action=processedTrueActionsBatched,  # true action at timestep do a whole batch
            )  #  predicted with true action input policies

            ##### Unbind by zero to iterate over the batch
            if self.config["discrete_actions"]:
                # softmax_tensors = predictedPoliciesForTrueAction[2]
                softmax_tensors = predictedPoliciesForTrueAction[
                    2
                ]  # torch.unbind(, dim=1)[0]

                # batch size, other agent num, policy size

                batch_predicted_true_dist_i = [
                    softmax_tensors
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor

                batch_predicted_true_dist_i = torch.stack(
                    batch_predicted_true_dist_i, dim=0
                )  # counterfactuals, batch size, other agent num, policy size

                # print("softmax_tensors", softmax_tensors.shape)
                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape)

                batch_predicted_true_dist_i = batch_predicted_true_dist_i.permute(
                    1, 0, 2, 3
                )

                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape) # batch size, counterfactuals, other agent num, policy size

                MyTrueActions = processedTrueActionsBatched.squeeze(1)
                allPossibleActions = torch.arange(
                    1, self.config["social_influence_n_counterfactuals"] + 2
                )
                BatchCounterfactualActions = []
                for true_action in MyTrueActions:
                    counterfactual_actions = allPossibleActions[
                        allPossibleActions != true_action
                    ]
                    BatchCounterfactualActions.append(counterfactual_actions)

                # no need to permute to get batch size, counterfactual, action dim
                BatchCounterfactualActionsForThisBatchOfAction = torch.concat(
                    BatchCounterfactualActions, dim=0
                ).float()
                # unsqueeze last dimension
                BatchCounterfactualActionsForThisBatchOfAction = (
                    BatchCounterfactualActionsForThisBatchOfAction.unsqueeze(-1)
                )
                # print("BatchCounterfactualActionsForThisBatchOfAction", BatchCounterfactualActionsForThisBatchOfAction.shape)

            else:
                predicted_true_mu_i = torch.unbind(
                    predictedPoliciesForTrueAction[0], dim=1
                )[0]
                predicted_true_std_i = torch.unbind(
                    predictedPoliciesForTrueAction[1], dim=1
                )[
                    0
                ]  # get per agent since this is a tuple

                batch_predicted_true_mu_i = [
                    predicted_true_mu_i
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor
                batch_predicted_true_mu_i = torch.stack(
                    batch_predicted_true_mu_i, dim=0
                ).permute(1, 0, 2)
                batch_predicted_true_std_i = [
                    predicted_true_std_i
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]  # produces a duplicate of the same tensor
                batch_predicted_true_std_i = torch.stack(
                    batch_predicted_true_std_i, dim=0
                ).permute(1, 0, 2)

                batch_predicted_true_dist_i = Normal(
                    batch_predicted_true_mu_i, batch_predicted_true_std_i
                )

                # unbind along the batch dimension
                # [bs, count_num, action_dim]
                BatchCounterfactualActionsForThisBatchOfAction = [
                    torch.randn_like(processedTrueActionsBatched)
                    for i in range(self.config["social_influence_n_counterfactuals"])
                ]
                BatchCounterfactualActionsForThisBatchOfAction = (
                    torch.concat(BatchCounterfactualActionsForThisBatchOfAction, dim=0)
                    
                )
            # [bs, count_num, obs_dim]
            Batch_true_obs_batchList = [
                processedObsBatched
                for i in range(self.config["social_influence_n_counterfactuals"])
            ]
            Batch_true_obs_batchList = torch.concat(
                Batch_true_obs_batchList, dim=0
            )

            predictedPoliciesForCounterFactualAction = self.social_influence_networks[
                agent_id
            ](
                x=Batch_true_obs_batchList,  # batch size, counterfactuals, obs size
                action=BatchCounterfactualActionsForThisBatchOfAction,  # this is a single action with batchsize 1 # batch size, counterfactuals, action size
            )

            # batch size, other agents, counterfactual size, policy size

            ## JS divergence
            if self.config["discrete_actions"]:
                # predicted_counterfactual_dist = torch.unbind(predictedPoliciesForCounterFactualAction[2], dim=1)[0]
                predicted_counterfactual_dist = torch.reshape(
                    predictedPoliciesForCounterFactualAction[2],
                    (
                        -1,
                        self.config["social_influence_n_counterfactuals"],
                        self.num_agents - 1,
                        predictedPoliciesForCounterFactualAction[2].shape[-1],
                    ),
                )
                # print("predicted_counterfactual_dist", predicted_counterfactual_dist.shape) # batch size, counterfactual size, other agents, policy size
                # print("batch_predicted_true_dist_i", batch_predicted_true_dist_i.shape) # batch size, counterfactual size, other agents, policy size
                # assert that valid probability distributions (i.e., they are non-negative and sum to 1)
                assert (
                    predicted_counterfactual_dist.shape
                    == batch_predicted_true_dist_i.shape
                ), "Predicted counterfactual dist and batch predicted true dist should have the same shape"
                # assert (predicted_counterfactual_dist >= 0).all(), "Predicted counterfactual dist should be non-negative"
                # assert (torch.abs(predicted_counterfactual_dist.sum(dim=-1) -1 ) < 1e-6).all(), f"Predicted counterfactual dist should sum to 1, actual sum is {predicted_counterfactual_dist.sum(dim=-1)}"
                # assert (batch_predicted_true_dist_i >= 0).all(), "Batch predicted true dist should be non-negative"
                # assert (torch.abs(batch_predicted_true_dist_i.sum(dim=-1) -1 ) < 1e-6).all(), f"Batch predicted true dist should sum to 1, actual sum is {batch_predicted_true_dist_i.sum(dim=-1)}"

                LOGITS = False
                if LOGITS:
                    #### This is for KL divergence
                    predNormDist = Categorical(
                        logits=predicted_counterfactual_dist
                    )  # this one should be logprobs because we forward with logits
                    trueNormDist = Categorical(
                        logits=batch_predicted_true_dist_i
                    )  # This one should be logprobs because we prep data above

                    # Convert logits to probabilities
                    # true_probs = F.softmax(batch_predicted_true_dist_i, dim=-1)
                    # pred_probs = F.softmax(predicted_counterfactual_dist, dim=-1)
                    true_probs = torch.exp(batch_predicted_true_dist_i)
                    pred_probs = torch.exp(predicted_counterfactual_dist)
                else:
                    predicted_counterfactual_dist = signal_process_remove_low_probs(
                        predicted_counterfactual_dist
                    )
                    batch_predicted_true_dist_i = signal_process_remove_low_probs(
                        batch_predicted_true_dist_i
                    )

                    predNormDist = Categorical(
                        probs=predicted_counterfactual_dist
                    )  # this one should be logprobs because we forward with logits
                    trueNormDist = Categorical(
                        probs=batch_predicted_true_dist_i
                    )  # This one should be logprobs because we prep data above

                    true_probs = batch_predicted_true_dist_i
                    pred_probs = predicted_counterfactual_dist

                mixture_dist = 0.5 * (true_probs + pred_probs)
                # mixture_dist = mixture_dist / mixture_dist.sum(dim=-1, keepdim=True) # normalize
                # mixture_dist_logits = F.log_softmax(mixture_dist, dim=-1)
                # mixtureDist = Categorical(logits=mixture_dist_logits)
                mixtureDist = Categorical(probs=mixture_dist)

                js_divergence = 0.5 * (
                    kl.kl_divergence(trueNormDist, mixtureDist)
                    + kl.kl_divergence(predNormDist, mixtureDist)
                )

                # js divergence is non-negative so clamp
                js_divergence = torch.clamp(js_divergence, min=0)

                # singleLoss = js_divergence.mean()

                # js_divergence = 0.5 * ((kl.kl_divergence(Categorical(probs=batch_predicted_true_dist_i), Categorical(probs=mixture_dist)) + kl.kl_divergence(Categorical(probs=predicted_counterfactual_dist), Categorical(probs=mixture_dist))))

                # TODO put this back
                # assert (js_divergence >= 0).all(), f"JS divergence should be non-negative but it is {js_divergence}"
            else:
                predicted_counterfactual_mu_i = torch.unbind(
                    predictedPoliciesForCounterFactualAction[0], dim=1
                )[0]
                predicted_counterfactual_std_i = torch.unbind(
                    predictedPoliciesForCounterFactualAction[1], dim=1
                )[0]
                predicted_counterfactual_dist = Normal(
                    predicted_counterfactual_mu_i, predicted_counterfactual_std_i
                )
                kl_divergence = kl.kl_divergence(
                    batch_predicted_true_dist_i, predicted_counterfactual_dist
                )
                print("TODO change this to js divergence for continuous actions")
                # TODO contd broken for now
                assert (
                    kl_divergence >= 0
                ), "KL divergence should be non-negative but it is {kl_divergence}"

            ###################### The change, do not mean and variance for the counterfactuals, just send it back
            social_influence_metric = js_divergence
            counterfactuals = predicted_counterfactual_dist

            # TODO padding for the counterfactuals? - for dead agents - where each counterfactuals should be zero

            # TODO this is only for discrete
            softmax_tensors_predictedPoliciesForTrueAction = (
                predictedPoliciesForTrueAction[2]
            )

            # TODO zero

            return (
                social_influence_metric,
                counterfactuals,
                softmax_tensors_predictedPoliciesForTrueAction,
            )
            # # assert len(social_influence_reward) == processedObsBatched.shape[0]

            # # Assign the first dimension of individualTrueAction to batchsize
            # batchsize = individualTrueAction.shape[0]            # Calculate the number of zeroes needed to pad social_influence_reward to the desired length
            # padding_length = batchsize - social_influence_reward.size(0)

            # # Create a tensor of zeroes with the required size
            # zero_padding = torch.zeros((padding_length,) + social_influence_reward.shape[1:])

            # # Concatenate social_influence_reward and the tensor of zeroes along dimension 0
            # social_influence_reward_padded = torch.cat((social_influence_reward, zero_padding), dim=0)

            # social_influence_variance_padded = torch.cat((social_influence_variance, zero_padding), dim=0)

            # #assert there are no negatives in the reward
            # # assert (social_influence_reward_padded >= 0).all(), "Social influence reward should be non-negative, {social_influence_reward_padded}"

            # return social_influence_reward_padded, social_influence_variance_padded # this should be a list per timestep

    # def signal_process_remove_low_probs(self, probs):
    #     # remove low probabilities

    #     # dynamicThreshold is mean of probs - 2 stddev
    #     dynamicThreshold = probs.mean(dim=-1) - 2 * probs.std(dim=-1)
    #     probs = torch.where(probs < dynamicThreshold, torch.zeros_like(probs), probs)
    #     #renormalize so all probs add up to 1
    #     probs = probs / probs.sum(dim=-1, keepdim=True)
    #     return probs


class SocialInfluencePredictor(nn.Module):
    def __init__(
        self,
        n_obs,
        # n_skills,  # 4
        action_dim,
        otherAgentPolicyDims,  # list of other agent action outputs
        # action_bounds=[-1, 1],
        n_hidden_units=256,
        num_agents=0,
        social_influence_n_counterfactuals=50,
        LOG_STD_MAX=2,
        LOG_STD_MIN=-20,
        device="cuda:0",
        layerNormEnabled=True,
        discrete_actions=True,
        si_loss_type="kl",
        only_use_argmax_policy=False,
        # lstm=False,
    ):
        super().__init__()
        self.device = device
        self.num_agents = num_agents
        self.n_other_agents = self.num_agents - 1
        self.action_dim = action_dim
        self.otherAgentPolicyDims = otherAgentPolicyDims
        self.n_obs = n_obs  # 11
        self.multi_agent_policy_output_size = self.action_dim * (self.n_other_agents)
        self.social_influence_n_counterfactuals = social_influence_n_counterfactuals
        self.layerNormEnabled = layerNormEnabled
        self.si_loss_type = si_loss_type
        self.only_use_argmax_policy = only_use_argmax_policy

        # Check if action dim is 1 and other agent policy dim is not 1, then we are likely using discrete actions and so should softmax
        self.DISCRETE_ACTIONS = discrete_actions

        # if DISCRETE_ACTIONS:
        #     action_bounds = [0, 1]
        # else:
        #     action_bounds = [-1, 1]

        # action_min, action_max = action_bounds

        #### Calculate obs_dim basic
        # n_skills  # observation should also include skill size
        # n_actions should be size of action, since one counterfactual action / true action is appended in

        ## Skill is taken out of observation as per (Take the skill out, because if everything else is the same, and the skill is changed, should things change in terms of whether the agents are coordinated?)
        # obs_dim = n_obs + n_skills + action_dim
        obs_dim = 129  # TODO Needs a way to calculate this

        self.hidden1 = nn.Linear(in_features=obs_dim, out_features=n_hidden_units)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Calculate the size of the flattened feature vector after convolutions and pooling
        # self.fc_input_dim = self._calculate_fc_input_dim()

        # self.fc1 = nn.Linear(self.fc_input_dim, 256)  # First fully connected layer
        # self.fc2 = nn.Linear(256, embedding_dim)

        init_weight(self.hidden1)
        init_weight(self.conv1)
        init_weight(self.conv2)
        init_weight(self.conv3)

        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(
            in_features=n_hidden_units,
            out_features=n_hidden_units,  # Note this is where it is different from
        )
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        if self.layerNormEnabled:
            self.layerNorm1 = nn.LayerNorm(n_hidden_units)
            self.layerNorm2 = nn.LayerNorm(n_hidden_units)
            self.LayerNorms = nn.ModuleList()

        self.HiddenNets = nn.ModuleList()
        self.MuNetworks = nn.ModuleList()
        self.LogStdNetworks = nn.ModuleList()
        for (
            otherAgent_single_agent_policy_input_size
        ) in otherAgentPolicyDims:  # range(self.n_other_agents):
            agent_hidden = nn.Linear(
                in_features=n_hidden_units,
                out_features=otherAgent_single_agent_policy_input_size,
            )
            init_weight(agent_hidden)
            agent_hidden.bias.data.zero_()
            if self.layerNormEnabled and otherAgent_single_agent_policy_input_size > 1:
                agent_hidden_LN = nn.LayerNorm(
                    otherAgent_single_agent_policy_input_size
                )
                self.LayerNorms.append(agent_hidden_LN)

            mu = nn.Linear(
                in_features=otherAgent_single_agent_policy_input_size,  # policy dim
                out_features=otherAgent_single_agent_policy_input_size,  # action_dim assumed to be same as policy dim
            )
            init_weight(mu, initializer="xavier uniform")
            mu.bias.data.zero_()

            log_std = nn.Linear(
                in_features=otherAgent_single_agent_policy_input_size,
                out_features=otherAgent_single_agent_policy_input_size,
            )
            init_weight(log_std, initializer="xavier uniform")
            log_std.bias.data.zero_()

            self.HiddenNets.append(agent_hidden)
            self.MuNetworks.append(mu)
            self.LogStdNetworks.append(log_std)

        self.LOG_STD_MAX = LOG_STD_MAX
        self.LOG_STD_MIN = LOG_STD_MIN

    def forward(self, x, action=None):
        # x should be obs + action
        x = x.squeeze()
        if len(x.shape) == 4:
            x = torch.permute(x, (0, -1, 1, 2))  # no counterfactuals
        # else:
        # print("X", x.shape)
        # print("action", action.shape)

        if action is not None:  # alr concatenated
            # check shape and if not correct adjust by squeezing both tensors
            # while len(action.shape) != len(x.shape):
            #     # if len(action.shape) == 3:
            #     #     action = action.squeeze(1)
            #     # x = x.squeeze()
            #     if len(action.shape) < 3:
            #         action.unsqueeze(1)
            #     if len(x.shape) < 3:
            #         x.unsqueeze(1)

            # x = torch.cat([x, action], dim=-1)
            
            # print("X", x.shape)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            # x = self.pool(x)
            # x = torch.flatten(x)
            x = x.squeeze()
            # print(x.shape)
            # print(action.shape)
            # print("X", x.shape)
            # print("action", action.shape)
            action = action.squeeze()
            if len(action.shape) == 1:
                action = action.unsqueeze(1)
            x = torch.concat([x, action], dim=-1)
            # print(x.shape)
        if self.layerNormEnabled:
            x = F.relu(self.layerNorm1(self.hidden1(x)))
            x = F.relu(
                self.layerNorm2(self.hidden2(x))
            )  # may want to remove this as it is a bit extra can just do one hidden here and use other agent hidden below
        else:
            x = F.relu(self.hidden1(x))
            x = F.relu(self.hidden2(x))

        if self.layerNormEnabled:
            HdistList = [
                F.relu(ln(agent_hidden_net(x)))
                for agent_hidden_net, ln in zip(self.HiddenNets, self.LayerNorms)
            ]
        else:
            HdistList = [
                F.relu(agent_hidden_net(x)) for agent_hidden_net in self.HiddenNets
            ]

        if self.DISCRETE_ACTIONS:
            distList = [
                x - x.max() for x in HdistList
            ]  # Subtract the maximum value for numerical stability
            # distList = [x + 1e-8 for x in distList] # Add a small epsilon /value to prevent model divergence
            # distList = [F.log_softmax(x, dim=-1) for x in distList]
            distList = [F.softmax(x, dim=-1) for x in distList]
            distList = [
                x + 1e-8 for x in distList
            ]  # Add a small epsilon value to prevent model divergence

            # distList = [x + 1e-8 for x in distList] # Add a small epsilon value to prevent model divergence
            # Normalize again to make the sum of probabilities equal to 1

            # distList = [x / x.sum() for x in distList] # Normalize again to make the sum of probabilities equal to 1
            # print("distList first", distList[0].shape)

            # Check that the distributions are valid, sum to one
            # for dist in distList:
            #     # print("dist", dist.shape)
            #     assert (dist >= 0).all(), f"Predicted distribution should be non-negative, but it is {dist}"
            #     assert (torch.abs(dist.sum(dim=-1) - 1) < 1e-6).all(), f"Predicted distribution should sum to 1, but it is {dist.sum(dim=-1)}"
            return None, None, torch.stack(distList, dim=1)
        else:
            MuList, LogStdList = zip(
                *[
                    (self.MuNetworks[i](dist), self.LogStdNetworks[i](dist))
                    for i, dist in enumerate(HdistList)
                ]
            )

            StdList = [
                log_std.clamp(min=self.LOG_STD_MIN, max=self.LOG_STD_MAX).exp()
                for log_std in LogStdList
            ]

            distList = [Normal(mu, std) for mu, std in zip(MuList, StdList)]

            return (
                torch.stack(MuList, dim=1),
                torch.stack(StdList, dim=1),
                distList,
            )  # This list items are per agent, but due to the way we do batch, just stack. Later unstack.

    def calc_social_influence_loss(
        self,
        individualObs,
        individualTrueAction,
        ListOfTruePoliciesOfAgents,
        agent_to_calc_loss_for,
        AGENTS_CAN_DIE=False,
    ):
        """training social influence predictor only. Train predictor with actual agent actions and actual other agent policy for each agent"""

        # AgentCounter = 0
        social_influence_acc = 0

        AgentCounter = agent_to_calc_loss_for

        # within the batch of action, if action is noop (0) then delete this index from action and observation and ListOfTruePoliciesOfAgents (truncate)
        # Get the indices where the action is not 0
        non_noop_indices = np.where(individualTrueAction != 0)[0]

        # Select the corresponding elements from the action, observation, and ListOfTruePoliciesOfAgents arrays
        if AGENTS_CAN_DIE:
            # Get the indices where the action is not 0
            non_noop_indices = np.where(individualTrueAction != 0)[0]

            # Select the corresponding elements from the action, observation, and ListOfTruePoliciesOfAgents arrays
            processedTrueActionsBatched = (
                individualTrueAction[non_noop_indices].unsqueeze(1).clone()
            )
            processedObsBatched = individualObs[non_noop_indices].unsqueeze(1).clone()
            OtherAgentTruePolicies = ListOfTruePoliciesOfAgents[
                non_noop_indices, :, :
            ].copy()
            if isinstance(OtherAgentTruePolicies, list):
                OtherAgentTruePolicies = [
                    x[:AgentCounter] + x[AgentCounter + 1 :]
                    for x in OtherAgentTruePolicies
                ]
            else:
                OtherAgentTruePolicies = np.delete(
                    OtherAgentTruePolicies, AgentCounter, axis=1
                )
            # Make this a torch tensor
            OtherAgentTruePolicies = torch.tensor(
                OtherAgentTruePolicies, dtype=torch.float32
            )

            # Add small epsilon to prevent model divergence
            OtherAgentTruePolicies = (
                OtherAgentTruePolicies + 1e-8
            )  # This is typically done to prevent division by zero or taking the logarithm of zero. However, in this case, it's not necessary and can be removed. The log_softmax function is already designed to be numerically stable and won't cause issues with zero values.

            # Do log softmax
            # OtherAgentTruePolicies = F.log_softmax(OtherAgentTruePolicies, dim=-1)

            # Normalize again to make the sum of probabilities equal to 1
            # OtherAgentTruePolicies = OtherAgentTruePolicies / OtherAgentTruePolicies.sum(/dim=-1, keepdim=True)

        else:
            processedTrueActionsBatched = individualTrueAction.clone()
            processedObsBatched = individualObs.clone()
            OtherAgentTruePolicies = ListOfTruePoliciesOfAgents.copy()

            if isinstance(OtherAgentTruePolicies, list):
                OtherAgentTruePolicies = [
                    x[:AgentCounter] + x[AgentCounter + 1 :]
                    for x in OtherAgentTruePolicies
                ]
            else:
                OtherAgentTruePolicies = np.delete(
                    OtherAgentTruePolicies, AgentCounter, axis=1
                )

            OtherAgentTruePolicies = torch.tensor(
                OtherAgentTruePolicies, dtype=torch.float32
            )

        # individualObs = torch.tensor(individualObs, dtype=torch.float32)
        # individualTrueAction = torch.tensor(individualTrueAction, dtype=torch.float32)

        # # detect nan or inf
        # if torch.isnan(individualObs).any() or torch.isnan(individualTrueAction).any() or torch.isnan(OtherAgentTruePolicies).any():
        #     print(f"WARNING Nan in obs {torch.isnan(individualObs).any()} or actions {torch.isnan(individualTrueAction).any()} or policies {torch.isnan(OtherAgentTruePolicies).any()}")

        # if torch.isinf(individualObs).any() or torch.isinf(individualTrueAction).any() or torch.isinf(OtherAgentTruePolicies).any():
        #     print(f"WARNING Inf in obs {torch.isinf(individualObs).any()} or actions {torch.isinf(individualTrueAction).any()} or policies {torch.isinf(OtherAgentTruePolicies).any()}")

        predictedPoliciesForTrueAction = self(
            x=processedObsBatched,
            action=processedTrueActionsBatched,
        )  # Returns

        if self.DISCRETE_ACTIONS:
            _, _, distListStack = predictedPoliciesForTrueAction
            predictedPolicies = torch.unbind(distListStack, dim=1)
        else:
            MuList, StdList, distList = predictedPoliciesForTrueAction
            MuList = torch.unbind(MuList, dim=1)
            StdList = torch.unbind(StdList, dim=1)
            predictedPolicies = list(zip(MuList, StdList, distList))

        # Do Individual policy comparison
        OtherAgentTruePolicies = torch.unbind(OtherAgentTruePolicies, dim=1)
        for (
            predOtherAgentPolicy,
            trueOtherAgentPolicy,
        ) in zip(predictedPolicies, OtherAgentTruePolicies):
            if self.DISCRETE_ACTIONS:
                # This is softmax instead of normal but should be fine
                # Wrap the distributions in a Categorical

                predOtherAgentPolicy = predOtherAgentPolicy.squeeze(1)
                # print("predOtherAgentPolicy", predOtherAgentPolicy.shape)
                # print("trueOtherAgentPolicy", trueOtherAgentPolicy.shape)
                assert predOtherAgentPolicy.shape == trueOtherAgentPolicy.shape

                # print to check for infs
                if (
                    torch.isnan(predOtherAgentPolicy).any()
                    or torch.isnan(trueOtherAgentPolicy).any()
                ):
                    print(
                        f"WARNING Nan in predOtherAgentPolicy {torch.isnan(predOtherAgentPolicy).any()} or trueOtherAgentPolicy {torch.isnan(trueOtherAgentPolicy).any()}"
                    )
                if (
                    torch.isinf(predOtherAgentPolicy).any()
                    or torch.isinf(trueOtherAgentPolicy).any()
                ):
                    print(
                        f"WARNING Inf in predOtherAgentPolicy {torch.isinf(predOtherAgentPolicy).any()} or trueOtherAgentPolicy {torch.isinf(trueOtherAgentPolicy).any()}"
                    )

                # reshape for batch size to add a dimension
                # predOtherAgentPolicy = predOtherAgentPolicy.unsqueeze(1)
                # trueOtherAgentPolicy = trueOtherAgentPolicy.unsqueeze(1)

                #### This is for KL divergence
                # predNormDist = Categorical(logits=predOtherAgentPolicy) # this one should be logprobs because we forward with logits
                predNormDist = Categorical(
                    probs=predOtherAgentPolicy
                )  # this one should be probs because we forward with softmax
                trueNormDist = Categorical(
                    probs=trueOtherAgentPolicy
                )  # This one should be logprobs because we prep data above

                if self.si_loss_type == "kl":
                    singleLoss = kl.kl_divergence(predNormDist, trueNormDist).mean()
                elif self.si_loss_type == "js":
                    # Convert logits to probabilities
                    true_probs = F.softmax(trueOtherAgentPolicy, dim=-1)
                    pred_probs = torch.exp(predOtherAgentPolicy)
                    # pred_probs = F.softmax(predOtherAgentPolicy, dim=-1)

                    mixture_dist = 0.5 * (true_probs + pred_probs)

                    mixtureDist = Categorical(probs=mixture_dist)

                    js_divergence = 0.5 * (
                        (
                            kl.kl_divergence(trueNormDist, mixtureDist)
                            + kl.kl_divergence(predNormDist, mixtureDist)
                        )
                    )

                    singleLoss = torch.clamp(js_divergence, min=0).mean()
                elif self.si_loss_type == "bce":
                    # binary cross entropy should get a one hot encoding of the true action for trueOtherAgentPolicy
                    singleLoss = F.binary_cross_entropy(
                        predOtherAgentPolicy, trueOtherAgentPolicy
                    )
                else:
                    print("WARNING: Invalid social influence loss type")

            else:
                # predMu, predStd, predNormDist = predOtherAgentPolicy
                # trueMu, trueStd, trueNormDist = trueOtherAgentPolicy
                # Already a normal
                predMu, predStd = predOtherAgentPolicy
                predNormDist = [Normal(mu, std) for mu, std in zip(predMu, predStd)]

                trueNormDist = trueOtherAgentPolicy

                myLoss = 0
                for i in predNormDist:
                    for j in trueNormDist:
                        myLoss += kl.kl_divergence(i, j).mean()

                singleLoss = myLoss / len(predNormDist)  # average it

            social_influence_acc += singleLoss

            # print("singleLoss", singleLoss)

        # Since we have self.num_agents to compute, self.num_agents - 1 policies comparisons per agent
        adjustedSocialInfluenceLoss = social_influence_acc / (
            (self.num_agents - 1) * self.num_agents
        )

        # print("adjustedSocialInfluenceLoss", adjustedSocialInfluenceLoss)

        return adjustedSocialInfluenceLoss
