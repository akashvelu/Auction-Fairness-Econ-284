import torch
import numpy as np
from torch.distributions import Normal


class PolicyNetwork(torch.nn.Module):
    def __init__(self, in_features, reserve_price, hidden_size=32, init_std=0.5):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        self.reserve_price = reserve_price
        self.log_std = torch.nn.Parameter(torch.log(torch.from_numpy(np.array([init_std]))))
        self.log_std.requires_grad = True

    def forward(self, x):
        hidden = torch.nn.functional.relu(self.fc1(x))
        out = torch.exp(self.fc2(hidden))
        mean = out + self.reserve_price
        dist = Normal(loc=mean, scale=torch.exp(self.log_std))
        return dist

    def get_action(self, state):
        # state should be B, D
        dist = self.forward(state)
        sample = dist.sample()
        mode = dist.mean
        sample_log_prob = dist.log_prob(sample)
        mode_log_prob = dist.log_prob(mode)
        return sample, mode, sample_log_prob, mode_log_prob
