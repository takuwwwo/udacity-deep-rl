import random
import pickle

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment, BrainInfo

from ddpg import ddpg, DDPGAgent
from ppo import ppo, PPOAgent
from trpo import trpo, TRPOAgent
from settings import *


def main():
    # setup environment
    env = UnityEnvironment(
        file_name="./Reacher_Linux/Reacher.x86_64")
    # env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # set seed
    torch.manual_seed(0)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size

    device = torch.device('cpu')

    if ALGORITHM == 'DDPG':
        agent = DDPGAgent(state_size, action_size, HIDDEN_UNITS, SEED, EPS_START, device)
        scores = ddpg(agent, env, brain_name, n_episodes=100)
        with open('result_ddpg.pkl', 'wb') as f:
            pickle.dump(scores, f)
    elif ALGORITHM == 'TRPO':
        agent = TRPOAgent(state_size, action_size, HIDDEN_UNITS, SEED, EPS_START, device)
        scores = trpo(agent, env, brain_name, n_episodes=100)
        with open('result_trpo.pkl', 'wb') as f:
            pickle.dump(scores, f)
    elif ALGORITHM == 'PPO':
        agent = PPOAgent(state_size, action_size, HIDDEN_UNITS, SEED, EPS_START, device)
        scores = ppo(agent, env, brain_name, n_episodes=100)
        with open('result_ppo.pkl', 'wb') as f:
            pickle.dump(scores, f)
    else:
        raise ValueError('Algorithm should be selected from DDPG, TRPO or PPO.')

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


if __name__ == '__main__':
    main()
