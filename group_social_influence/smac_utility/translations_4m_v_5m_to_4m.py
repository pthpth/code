import numpy as np
import torch 

def allThingsTranslationPadding(item, TypeDeclaration=None):
    # decipher type through last input size 
    
    LastInputSize = item.shape[-1]
    if TypeDeclaration is not None:
        LastInputSize = TypeDeclaration
    else:
        if LastInputSize == 88:
            LastInputSize = "observation"
        elif LastInputSize == 10:
            LastInputSize = "policy" # could be available actions
        elif LastInputSize == 111:
            LastInputSize = "state"
        else:
            raise ValueError("The last dimension of the input should be 88, 10 or 111, instead got {}".format(LastInputSize))
    
    if LastInputSize == "observation":
        return observation_translation_padding(item)
    elif LastInputSize == "policy":
        return policy_translation_padding(item)
    elif LastInputSize == "avail_action": # available action
        return available_actions_translation_padding(item)
    elif LastInputSize == "state":
        return state_translation_padding(item)
    
def allThingsTranslationChopping(item, TypeDeclaration=None):
    # decipher type through last input size 
    
    LastInputSize = item.shape[-1]
    if TypeDeclaration is not None:
        LastInputSize = TypeDeclaration
    else:
        if LastInputSize == 97:
            LastInputSize = "observation"
        elif LastInputSize == 11:
            LastInputSize = "policy"
        elif LastInputSize == 123:
            LastInputSize = "state"
        else:
            raise ValueError("The last dimension of the input should be 97, 11 or 123, instead got {}".format(LastInputSize))
    
    if LastInputSize == "observation":
        return observation_translation_chopping(item)
    elif LastInputSize == "policy":
        return policy_translation_chopping(item)
    elif LastInputSize == "avail_action": # available action
        return available_actions_translation_chopping(item)
    elif LastInputSize == "state":
        return state_translation_chopping(item)
    
    


def observation_translation_padding(smaller_observation):
    # There should be batchsize observation of size 89
    OriginalShape = smaller_observation.shape
    assert OriginalShape[-1] == 88, "Input should be 4m observation" # episode_length, n_rollout_threads, num_agents, observation_dim
    
    indiciesAtObsDim = [15, 30, 45, 65, 65, 65, 65, 65, 84]
    OriginallyNumpy = True
    if type(smaller_observation) == torch.Tensor:
        OriginallyNumpy = False
        smaller_observation = smaller_observation.numpy()
    
    larger_observation = np.insert(smaller_observation, indiciesAtObsDim, 0, axis=-1)

    assert larger_observation.shape[:-1] == OriginalShape[:-1], "The first few dimensions of the output should match the original shape (obs_padding), instead got {} and {}".format(larger_observation.shape, OriginalShape)   
    assert larger_observation.shape[-1] == 97, "Output should be 4m_v_5m observation (obs_padding), instead got {}".format(larger_observation.shape) 
    
    if OriginallyNumpy:
        return larger_observation
    else: 
        return torch.tensor(larger_observation)

def observation_translation_chopping(larger_observation):
    OriginalShape = larger_observation.shape
    assert OriginalShape[-1] == 97, "Input should be 4m_v_5m observation" # episode_length, n_rollout_threads, num_agents, observation_dim
    
    OriginallyNumpy = True
    if type(larger_observation) == torch.Tensor:
        OriginallyNumpy = False
        larger_observation = larger_observation.numpy()

    chopped_observation = np.delete(larger_observation, [15, 31, 47, 68, 69, 70, 71, 72, 92], axis=-1)
    
    assert chopped_observation.shape[:-1] == OriginalShape[:-1], "The first few dimensions of the output should match the original shape (obs_chopping)"
    assert chopped_observation.shape[-1] == 88, "Output should be 4m observation (obs_chopping)"
    if OriginallyNumpy:
        return chopped_observation
    else:
        return torch.tensor(chopped_observation)

def action_translation_padding(smaller_action):
    OriginalShape = smaller_action.shape
    assert OriginalShape[-1] == 10, "Input should be 4m action" # episode_length, n_rollout_threads, num_agents, action_dim
    
    indiciesAtActDim = [10] # The last element of the action
    
    larger_action = np.insert(smaller_action, indiciesAtActDim, 0, axis=-1)
    assert larger_action.shape[-1] == 11, "Output should be 4m_v_5m action"
    # assert larger_action.shape == (OriginalShape[0], OriginalShape[1], OriginalShape[2], 11), "Output should be 4m_v_5m action"
    
    return larger_action

def action_translation_chopping(larger_action):
    OriginalShape = larger_action.shape
    assert OriginalShape[-1] == 11, "Input should be 4m_v_5m action" # episode_length, n_rollout_threads, num_agents, action_dim
    chopped_action = np.delete(larger_action, -1, axis=-1)
    
    return chopped_action

def available_actions_translation_padding(smaller_available_actions):
    OriginalShape = smaller_available_actions.shape
    assert OriginalShape[-1] == 10, "Input should be 4m available actions"
    
    # Create a new array instead of modifying the input
    larger_available_actions = np.zeros(OriginalShape[:-1] + (11,), dtype=smaller_available_actions.dtype)
    larger_available_actions[..., :-1] = smaller_available_actions
    
    assert larger_available_actions.shape[-1] == 11, "Output should be 4m_v_5m available actions"
    # Shape of the output should be the same as the input, except the last dimension should be 11
    ExpectedShape = list(OriginalShape)
    ExpectedShape[-1] = 11
    assert larger_available_actions.shape == tuple(ExpectedShape), "Output should be 4m_v_5m available actions, instead got {}".format(larger_available_actions.shape)
    
    return larger_available_actions

def available_actions_translation_chopping(larger_available_actions):
    OriginalShape = larger_available_actions.shape
    assert OriginalShape[-1] == 11, "Input should be 4m_v_5m available actions"
    
    smaller_available_actions = larger_available_actions[..., :-1]
    assert smaller_available_actions.shape[-1] == 10, "Output should be 4m available actions"
    
    ExpectedShape = list(OriginalShape)
    ExpectedShape[-1] = 10
    assert smaller_available_actions.shape == tuple(ExpectedShape), "Output should be 4m available actions, instead got {}".format(smaller_available_actions.shape)
    
    return smaller_available_actions

def state_translation_padding(smaller_state):
    OriginalShape = smaller_state.shape
    assert OriginalShape[-1] == 111, "Input should be 4m state" # episode_length, n_rollout_threads, num_agents, state_dim
    
    indiciesAtStateDim = [18, 36, 54, 89, 89, 89, 89, 89, 89, 89, 89, 109] 
    larger_state = np.insert(smaller_state, indiciesAtStateDim, 0, axis=-1)
    
    larger_state = np.insert(smaller_state, indiciesAtStateDim, 0, axis=-1)
    assert larger_state.shape[-1] == 123, "Output should be 4m_v_5m state, instead got {}".format(larger_state.shape)
    
    return larger_state

def state_translation_chopping(larger_state):
    OriginalShape = larger_state.shape
    assert OriginalShape[-1] == 123, "Input should be 4m_v_5m state" # episode_length, n_rollout_threads, num_agents, state_dim
    chopped_state = np.delete(larger_state, [18, 37, 56, 92, 93, 94, 95, 96, 97, 98, 99,118], axis=-1)
    
    return chopped_state

def policy_translation_padding(smaller_policy):
    OriginalShape = smaller_policy.shape
    assert OriginalShape[-1] == 10, "Input should be 4m policy" # episode_length, n_rollout_threads, num_agents, action_dim
    
    if type(smaller_policy) == torch.Tensor:
        # create a tensor of zeros with the same shape as smaller_policy, except the last dimension is 1
        zeros = torch.zeros(*OriginalShape[:-1], 1, dtype=smaller_policy.dtype, device=smaller_policy.device)
        # concatenate smaller_policy and zeros along the last dimension
        larger_policy = torch.cat((smaller_policy, zeros), dim=-1)
    else:
        indiciesAtActDim = [10] # The last element of the action
        larger_policy = np.insert(smaller_policy, indiciesAtActDim, 0, axis=-1)
        
    assert larger_policy.shape[-1] == 11, "Output should be 4m_v_5m policy"
    # assert larger_action.shape == (OriginalShape[0], OriginalShape[1], OriginalShape[2], 11), "Output should be 4m_v_5m action"
    
    return larger_policy


def policy_translation_chopping(larger_policy):
    
    OriginalShape = larger_policy.shape
    assert OriginalShape[-1] == 11, "Input should be 4m_v_5m policy" # episode_length, n_rollout_threads, num_agents, action_dim

    # check if type is torch or numpy
    if type(larger_policy) == torch.Tensor:
        # PyTorch doesn't have a direct equivalent to np.delete, but you can achieve the same result with slicing and concatenation
        chopped_policy = torch.cat((larger_policy[..., :-1],), dim=-1)
        # need to make sure this is softmaxed
        chopped_policy = torch.exp(chopped_policy) / torch.sum(torch.exp(chopped_policy), dim=-1, keepdim=True)
    else:
        chopped_policy = np.delete(larger_policy, -1, axis=-1)
    
        # need to make sure this is softmaxed
        chopped_policy = np.exp(chopped_policy) / np.sum(np.exp(chopped_policy), axis=-1, keepdims=True)
        assert np.allclose(np.sum(chopped_policy, axis=-1), 1, atol=1e-7), "The output policy should be softmaxed - with some epsilon for error, instead got {}".format(np.sum(chopped_policy, axis=-1))    
    return chopped_policy



    
# if this file is directly run, then run the following test
if __name__ == "__main__":
    # Test the translation functions
    TeamA = np.array([[[0., 1., 1., 1., 1., 1., 0., 0., 1., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 1., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 0., 0.]],

    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 1., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 0., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
    [0., 1., 1., 1., 1., 1., 1., 1., 1., 0.]],

    [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 1., 0., 1., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])
    
    TeamB = np.array([[[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]],

    [[0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
    [0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.]]])
    
    print("TeamA shape: ", TeamA.shape)
    print("TeamB shape: ", TeamB.shape)
    
    # Test for padding and chopping 
    APadToB = allThingsTranslationPadding(TeamA, "avail_action")
    print("APadToB shape: ", APadToB.shape)
    InvertBack = allThingsTranslationChopping(APadToB, "avail_action")
    print("InvertBack shape: ", InvertBack.shape)
    assert np.allclose(InvertBack, TeamA), "Output of padding and chopping is reversible"
    

    # smaller_observation = np.random.rand(10, 2, 3, 88)
    # larger_observation = observation_translation_padding(smaller_observation)
    # assert larger_observation.shape == (10, 2, 3, 97)
    
    # smaller_action = np.random.rand(10, 2, 3, 10)
    # larger_action = action_translation_padding(smaller_action)
    # assert larger_action.shape == (10, 2, 3, 11)
    
    # chopped_observation = observation_translation_chopping(larger_observation)
    # assert chopped_observation.shape == (10, 2, 3, 88)
    
    # chopped_action = action_translation_chopping(larger_action)
    # assert chopped_action.shape == (10, 2, 3, 10)
    
    # smaller_state = np.random.rand(10, 2, 3, 111)
    # larger_state = state_translation_padding(smaller_state)
    # assert larger_state.shape == (10, 2, 3, 123)
    
    # chopped_state = state_translation_chopping(larger_state)
    # assert chopped_state.shape == (10, 2, 3, 111)
    
    # smaller_policy = np.random.rand(10, 2, 3, 10)
    # larger_policy = policy_translation_padding(smaller_policy)
    # assert larger_policy.shape == (10, 2, 3, 11)
    
    # chopped_policy = policy_translation_chopping(larger_policy)
    # assert chopped_policy.shape == (10, 2, 3, 10)
    
    # # Tests for available actions
    # StartingShape = (10, 2, 3, 11)
    # SmallerShape = (10, 2, 3, 10)
    # larger_available_actions = np.random.rand(10, 2, 3, 11)
    # # set all the last values to zero
    # larger_available_actions[..., -1] = 0
    # smaller_available_actions = available_actions_translation_chopping(larger_available_actions)
    # assert smaller_available_actions.shape == SmallerShape, "output of chopping is correct size"
    
    # rev_larger_available_actions = available_actions_translation_padding(smaller_available_actions)
    # assert rev_larger_available_actions.shape == StartingShape , "output of padding is correct size"
    # assert np.allclose(rev_larger_available_actions, larger_available_actions), "output of padding is reversible from chopping"
    
    # StartingShape = (2, 3, 11)
    # SmallerShape = (2, 3, 10)
    # larger_available_actions = np.random.rand(2, 3, 11)
    # # set all the last values to zero
    # larger_available_actions[..., -1] = 0
    # smaller_available_actions = available_actions_translation_chopping(larger_available_actions)
    # assert smaller_available_actions.shape == SmallerShape, "output of chopping is correct size"
    
    # rev_larger_available_actions = available_actions_translation_padding(smaller_available_actions)
    # assert rev_larger_available_actions.shape == StartingShape , "output of padding is correct size"
    # assert np.allclose(rev_larger_available_actions, larger_available_actions), "output of padding is reversible from chopping"
    
    
    print("All tests passed!")