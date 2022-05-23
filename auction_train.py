import numpy as np
import torch
from torch.distributions import Normal
from auction import Auction


class PolicyNetwork(torch.nn.Module):
    def __init__(self, in_features, reserve_price, hidden_size=32, init_std=0.05):
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


class Trainer:
    def __init__(self, auction, policy, lr, num_train_steps, episodes_per_update, gamma=1):
        self.auction = auction
        self.policy = policy
        self.lr = lr
        self.num_train_steps = num_train_steps
        self.episodes_per_update = episodes_per_update
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(lr=lr, params=self.policy.parameters())

    def rollout_auction(self, use_greedy=False):
        states = self.auction.reset()
        done = False
        states_list, bids_list, logprob_list, rewards_list, dones_list = [states], [], [], [], [
            np.zeros((self.auction.n_teams, 1))]
        while not done:
            bids_sampled, bids_mean, sample_logprob, mean_logprob = self.policy.get_action(torch.from_numpy(states))
            if use_greedy:
                bids = bids_mean.detach().numpy()
                logprob = mean_logprob
            else:
                bids = bids_sampled.detach().numpy()
                logprob = sample_logprob
            states, rewards, team_dones, done = self.auction.step(bids.flatten())

            states_list.append(states)
            bids_list.append(bids)
            logprob_list.append(logprob)
            rewards_list.append(rewards)
            dones_list.append(team_dones)
        # H, N, D
        return np.stack(states_list), np.stack(bids_list), torch.stack(logprob_list), np.stack(rewards_list), np.stack(
            dones_list)


    def train(self):
        for i in range(self.num_train_steps):
            loss = self.train_step()
            print("loss at iteration " + str(i) + ": " + str(loss))
            _, bids, _, rewards, dones = self.rollout_auction(use_greedy=True)
            team_returns = rewards.sum(0).flatten()
            # print("Eval team bids: ", bids[:, :, 0].T)
            print("eval team returns: ", team_returns)
            print("")

    def train_step(self):
        # rollout
        rollouts = [self.rollout_auction() for _ in range(self.episodes_per_update)]
        # B*N, H, D shape
        logprobs_batch = collapse_horizon(torch.stack([rollout[2] for rollout in rollouts]))
        rewards_batch = collapse_horizon(np.stack([rollout[3] for rollout in rollouts]))
        # B*N, H + 1, D; ith horizon index indicates if the i+1th state is "out of bounds"
        dones_batch = collapse_horizon(np.stack([rollout[4] for rollout in rollouts]))
        rewards_to_go = self.get_rewards_to_go(rewards_batch, dones_batch[:, 1:])
        rewards_to_go = torch.from_numpy(rewards_to_go)

        loss = (-1 * rewards_to_go * logprobs_batch * torch.from_numpy(1 - dones_batch[:, :-1])).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_rewards_to_go(self, rewards, dones, normalize=False):
        # shape is BS, H, 1
        rewards_to_go = np.zeros_like(rewards)
        BS, H, _ = rewards_to_go.shape
        R = np.zeros((BS, 1))
        for i in reversed(range(H)):
            R = rewards[:, i] + self.gamma * (1 - dones[:, i]) * R
            rewards_to_go[:, i] = R
        mean = rewards_to_go.mean()
        std = rewards_to_go.std()
        if normalize:
            return (rewards_to_go - mean) / (std + 1e-9)
        else:
            return rewards_to_go

def collapse_horizon(arr):
    # B, H, N, D -> B*N, H, D
    B, H, N, D = arr.shape
    if type(arr) == np.ndarray:
        arr = np.moveaxis(arr, 2, 1)
        return arr.reshape(B * N, H, D)
    else:
        arr = torch.moveaxis(arr, 2, 1)
        return arr.reshape(B * N, H, D)


if __name__ == '__main__':

    auction = Auction(n_teams=2, n_players=8, player_values=[8, 7, 6, 5, 4, 3, 2, 1], players_per_team=4, reserve_price=0)
    policy = PolicyNetwork(auction.state_dim, auction.reserve_price)


    # states, bids, rewards, dones = rollout_auction(policy, auction)
    trainer = Trainer(auction, policy, 3e-3, 1000, 500)
    trainer.train()
    a = 2



