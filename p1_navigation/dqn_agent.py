import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from model import QNetwork
from settings import *


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, device, ddqn: bool = False, dueling: bool = False,
                 distributional: bool = False, atoms: int = 51, v_min: float = -25., v_max: float = 25.,
                 norm_clip: float = 10.):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, fc_units=HIDDEN_UNITS, dueling=dueling,
                                       distributional=distributional, atoms=atoms).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, fc_units=HIDDEN_UNITS, dueling=dueling,
                                        distributional=distributional, atoms=atoms).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # setup
        self.ddqn = ddqn
        self.distributional = distributional
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atoms - 1.)
        self.support = torch.linspace(v_min, v_max, atoms).to(device)  # Support (range) of z
        self.norm_clip = norm_clip

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            if self.distributional:
                action_values = torch.sum(action_values * self.support[None, None, :], dim=-1)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        rewards = torch.squeeze(rewards, 1)
        dones = torch.squeeze(dones, 1)
        batch = actions.shape[0]
        atoms = self.atoms

        if self.distributional:
            outputs = self.qnetwork_local(states)

            # local_q.shape == (B, Atoms)
            local_q = torch.gather(outputs, dim=1, index=actions[:, :, None].expand(batch, 1, atoms)).squeeze(1)
            if self.ddqn:
                with torch.no_grad():
                    next_local_outputs = self.qnetwork_local(next_states)  # shape == (B, A, Atoms)
                    next_e = torch.sum(self.support[None, None, :] * next_local_outputs, dim=-1)  # shape == (B, A)
                    next_best_actions = torch.argmax(next_e, dim=-1)  # next_best_actions.shape == (B)
                    next_target_outputs = self.qnetwork_target(next_states)  # shape == (B, A, Atoms)
                    next_targets = \
                        torch.gather(next_target_outputs, dim=1, index=next_best_actions[:, None, None].expand(
                            batch, 1, atoms)).squeeze(1)  # (B, Atoms)
            else:
                with torch.no_grad():
                    next_target_outputs = self.qnetwork_target(next_states)  # shape == (B, A, Atoms)
                    next_e = torch.sum(self.support[None, None, :] * next_target_outputs, dim=-1)  # shape == (B, A)
                    next_best_actions = torch.argmax(next_e, dim=-1)  # next_best_actions.shape == (B)
                    next_targets = \
                        torch.gather(next_target_outputs, dim=1, index=next_best_actions[:, None, None].expand(
                            batch, 1, atoms)).squeeze(1)  # (B, Atoms)

            t_z = rewards[:, None] + gamma * (1. - dones[:, None]) * self.support[None, :]  # t_z.shape == (B, Atoms)
            t_z = t_z.clamp(self.v_min, self.v_max)
            t_z = t_z.clamp(self.v_min, self.v_max)
            b = (t_z - self.v_min) / self.delta_z  # b.shape == (B, Atoms)
            lower = b.floor().to(torch.int64)  # upper.shape == (B, Atoms)
            upper = b.ceil().to(torch.int64)  # lower.shape == (B, Atoms)

            lower[(upper > 0) * (lower == upper)] -= 1
            upper[(lower < atoms - 1) * (lower == upper)] += 1

            m = torch.zeros((batch, atoms)).to(self.device)
            offset = torch.linspace(0, (batch - 1) * atoms, batch, dtype=torch.int64)[:, None].expand(batch, atoms).to(self.device)
            m.view(-1).index_add_(0, (lower + offset).view(-1), (next_targets * (upper.float() - b)).view(-1))
            m.view(-1).index_add_(0, (upper + offset).view(-1), (next_targets * (b - lower.float())).view(-1))

            loss = torch.mean(-torch.sum(m * torch.log(local_q), dim=1))
        else:
            outputs = self.qnetwork_local(states)
            local_q = torch.gather(outputs, dim=1, index=actions).squeeze(1)

            if self.ddqn:
                next_actions = torch.argmax(self.qnetwork_local(next_states), dim=1).unsqueeze(1).detach()
                next_q = torch.gather(self.qnetwork_target(next_states), dim=1, index=next_actions).squeeze(1)
                targets = gamma * next_q * (1 - dones) + rewards
            else:
                targets = gamma * torch.max(self.qnetwork_target(next_states), dim=1)[0] * (1 - dones) + rewards
            loss = torch.sum(torch.square(local_q - targets))
        self.qnetwork_local.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.norm_clip)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)