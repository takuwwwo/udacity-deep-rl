import torch
from typing import Tuple
from collections import deque

import numpy as np

from model import ActorNetwork, CriticNetwork
from settings import *
from replay_buffer import ReplayBuffer
from utils import hard_update, soft_update
import torch.optim as optim
from torch.distributions import MultivariateNormal


class PPOAgent:
    def __init__(self, input_size: int, action_size: int, hidden_size: int, seed: int, eps: float, device):
        self.actor_target = ActorNetwork(input_size, action_size, hidden_size)
        self.actor_local = ActorNetwork(input_size, action_size, hidden_size)
        hard_update(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)

        self.critic_target = CriticNetwork(input_size, action_size, hidden_size, evaluate_q=False)
        self.critic_local = CriticNetwork(input_size, action_size, hidden_size, evaluate_q=False)
        hard_update(self.critic_target, self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        self.eps = eps

        self.t_step = 0
        self.device = device

    def step(self, states, actions, rewards, next_states, dones, log_probs):
        for state, action, reward, next_state, done, log_prob in zip(states, actions, rewards,
                                                                     next_states, dones, log_probs):
            self.memory.add(state, action, reward, next_state, done, log_prob)

        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            for _ in range(NUM_UPDATES):
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)
                    soft_update(self.critic_target, self.critic_local, TAU)
                    soft_update(self.actor_target, self.actor_local, TAU)

    def act(self, states):
        states = torch.tensor(states).float().to(self.device)
        self.actor_local.eval()
        self.actor_local.forward(states, normalize=False).squeeze(0)
        with torch.no_grad():
            actions_mean = self.actor_local.forward(states, normalize=True).squeeze(0)
        self.actor_local.train()

        actions_var = torch.diag_embed(torch.ones_like(actions_mean) * (self.eps ** 2.))
        dist = MultivariateNormal(actions_mean, actions_var)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions.detach().numpy(), log_probs.detach().numpy()

    def learn(self, experiences: Tuple, gamma: float):
        states, actions, rewards, next_states, dones, log_probs = experiences
        rewards = torch.squeeze(rewards, 1)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-07)
        dones = torch.squeeze(dones, 1)
        log_probs = torch.squeeze(log_probs, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_target_q = self.critic_target.forward(next_states, next_actions).squeeze(1)
            target_q = rewards + gamma * next_target_q * (1. - dones)
            target_q = target_q.detach()

        current_q = self.critic_local.forward(states, actions).squeeze(1)
        critic_loss = torch.mean(torch.square(target_q - current_q))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        target_current_q = self.critic_target.forward(states).squeeze(1).detach()
        advantages = target_q - target_current_q
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-07)
        actions_for_policy = self.actor_local(states)
        actions_var = torch.diag_embed(torch.ones_like(actions_for_policy) * (self.eps ** 2.))
        dist = MultivariateNormal(actions_for_policy, actions_var)
        cur_log_probs = dist.log_prob(actions)
        ratios = torch.exp(cur_log_probs - log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


def ppo(agent: PPOAgent, env, brain_name, n_episodes: int = 2000, max_t: int = 1000, eps_end=0.1, eps_decay=0.995):
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    scores_window = deque(maxlen=100 // num_agents)
    score_list = list()

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        scores = np.zeros(num_agents)

        states = env_info.vector_observations
        for t in range(max_t):
            actions, log_probs = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, log_probs)
            states = next_states
            scores += rewards
            if np.any(dones):
                break

        score = np.mean(scores)
        scores_window.append(score)
        score_list.append(score)
        agent.eps = max(eps_end, eps_decay * agent.eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100 // num_agents,
                                                                                         np.mean(scores_window)))
    torch.save(agent.critic_local.state_dict(), 'ppo_critic_checkpoint.pth')
    torch.save(agent.actor_local.state_dict(), 'ppo_actor_checkpoint.pth')

    return score_list