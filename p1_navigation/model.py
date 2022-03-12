import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=64, dueling: bool = False, distributional: bool = False,
                 atoms: int = 51):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = spectral_norm(nn.Linear(fc_units, fc_units))
        self.dueling = dueling
        self.action_size = action_size
        self.atoms = atoms
        self.distributional = distributional
        if distributional:
            if dueling:
                self.fc3_v = nn.Linear(fc_units, atoms)
                self.fc3_q = nn.Linear(fc_units, action_size * atoms)
            else:
                self.fc3 = nn.Linear(fc_units, action_size * atoms)
        else:
            if dueling:
                self.fc3_v = nn.Linear(fc_units, 1)
                self.fc3_q = nn.Linear(fc_units, action_size)
            else:
                self.fc3 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.fc3_v(x)
            q = self.fc3_q(x)
            if self.distributional:
                advantage = (q - torch.mean(q, dim=1, keepdim=True)).view(-1, self.action_size, self.atoms)
                output = advantage + v[:, None, :]
                output = torch.softmax(output, dim=-1)
            else:
                advantage = q - torch.mean(q, dim=1, keepdim=True)
                output = advantage + v
            return output
        else:
            if self.distributional:
                output = torch.softmax(self.fc3(x).view(-1, self.action_size, self.atoms), dim=-1)
                return output
            else:
                return self.fc3(x)
