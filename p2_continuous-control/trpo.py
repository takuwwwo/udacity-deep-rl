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
from torch.autograd import Variable


def conjugate_gradient(hessian_product, b, nsteps, tol=1e-7):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    for _ in range(nsteps):
        alpha = torch.dot(r, p) / torch.dot(p, hessian_product(p))
        x = x + alpha * p
        r_new = r - alpha * hessian_product(p)

        beta = torch.dot(r_new, r_new) / torch.dot(r, r)
        p = r_new + beta * p
        r = r_new
        if torch.norm(r) < tol:
            break

    return x


class TRPOAgent:
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
        old_actions_mean = self.actor_local(states).detach()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    actions_mean = self.actor_local(states)
            else:
                actions_mean = self.actor_local(states)
            actions_var = torch.diag_embed(torch.ones_like(actions_mean) * (self.eps ** 2.))
            dist = MultivariateNormal(actions_mean, actions_var)
            cur_log_probs = dist.log_prob(actions)
            ratios = torch.exp(cur_log_probs - log_probs)
            policy_loss = -torch.mean(ratios * advantages)
            return policy_loss

        def get_kl():
            new_actions_mean = self.actor_local(states)
            kl = (self.eps ** 2. + torch.square(new_actions_mean - old_actions_mean)) / (self.eps ** 2.) - 0.5
            return torch.mean(kl)

        self.trpo_step(get_loss, get_kl, 0.01)

    def trpo_step(self, get_loss, get_kl, delta):
        grads = torch.autograd.grad(get_loss(), self.actor_local.parameters())
        policy_loss_grad = torch.cat([grad.view(-1) for grad in grads])

        def hessian_product(v):
            kl = get_kl().mean()
            grads = torch.autograd.grad(kl, self.actor_local.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.actor_local.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + 1e-02 * v

        s = conjugate_gradient(hessian_product, -policy_loss_grad, 3)
        beta = torch.sqrt(2 * delta / torch.dot(s, hessian_product(s)))
        full_update = beta * s

        params = self.get_policy_params()
        params = self.line_search(get_loss, get_kl, params, full_update, delta)
        self.set_policy_params(params)

    def line_search(self, get_loss, get_kl, params, full_update, delta, max_backtracks=3):
        loss = get_loss(True).data
        for (n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
            new_params = params + stepfrac * full_update
            self.set_policy_params(new_params)
            new_loss = get_loss(True).data
            actual_improve = loss - new_loss

            new_kl = get_kl()
            if actual_improve.item() > 0. and new_kl.item() <= delta:
                return new_params
        return new_params

    def set_policy_params(self, flat_params):
        prev_ind = 0
        for param in self.actor_local.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def get_policy_params(self):
        params = []
        for param in self.actor_local.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params


def trpo(agent: TRPOAgent, env, brain_name, n_episodes: int = 2000, max_t: int = 1000, eps_end=0.1, eps_decay=0.995):
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

    torch.save(agent.critic_local.state_dict(), 'trpo_critic_checkpoint.pth')
    torch.save(agent.actor_local.state_dict(), 'trpo_actor_checkpoint.pth')

    return score_list