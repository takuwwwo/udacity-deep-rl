import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in hidden layer
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state, normalize: bool = True):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        if normalize:
            return torch.tanh(x)  # make the output in [-1., 1.]
        else:
            return x


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size, evaluate_q: bool = True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNetwork, self).__init__()
        self.evaluate_q = evaluate_q
        self.fcs1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        if evaluate_q:
            self.fc2 = nn.Linear(hidden_size+action_size, hidden_size)
        else:
            self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = spectral_norm(nn.Linear(hidden_size, hidden_size))
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, state, action=None):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.leaky_relu(self.ln1(self.fcs1(state)))
        if self.evaluate_q:
            x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
