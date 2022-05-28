import numpy as np
import torch
from auction import Auction
from policy_network import PolicyNetwork
import os

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

        # pad the list with default values (dummy for all elements other than the dones, which is set to 1
        remaining_len = self.auction.n_players - len(bids_list)
        states_list += [np.zeros_like(states_list[0]) for _ in range(remaining_len)]
        bids_list += [np.zeros_like(bids_list[0]) for _ in range(remaining_len)]
        logprob_list += [torch.zeros_like(logprob_list[0]) for _ in range(remaining_len)]
        rewards_list += [np.zeros_like(rewards_list[0]) for _ in range(remaining_len)]
        dones_list += [np.ones_like(dones_list[0]) for _ in range(remaining_len)]

        return np.stack(states_list), np.stack(bids_list), torch.stack(logprob_list), np.stack(rewards_list), np.stack(
            dones_list)

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

    def get_rewards_to_go(self, rewards, dones, normalize=True):
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

    def train(self):
        team_returns_over_training = []
        player_allocations_over_training = []

        for i in range(self.num_train_steps):
            loss = self.train_step()
            print("loss at iteration " + str(i) + ": " + str(loss))
            _, bids, _, rewards, dones = self.rollout_auction(use_greedy=True)
            team_returns = rewards.sum(0).flatten()
            team_returns_over_training.append(team_returns)
            player_allocations_over_training.append(self.auction.player_assignments)
            # print("Eval team bids: ", bids[:, :, 0].T)
            print("eval team returns: ", team_returns)
            print("")
        return np.stack(team_returns_over_training), np.stack(player_allocations_over_training)

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
    player_values = [[6, 6, 6], [5, 5, 5], [4, 4, 4], [3, 3, 3], [2, 2, 2], [1, 1, 1]]
    n_teams = 3
    n_players = 6
    players_per_team = 2
    reserve_price = 0

    auction = Auction(n_teams=n_teams,
                      n_players=n_players,
                      player_team_values=player_values,
                      players_per_team=players_per_team,
                      reserve_price=reserve_price)
    policy = PolicyNetwork(auction.state_dim, auction.reserve_price)


    # states, bids, rewards, dones = rollout_auction(policy, auction)
    trainer = Trainer(auction, policy, 5e-4, 1000, 1000)
    team_returns, player_allocations = trainer.train()

    scenario_name = "auction_nteams{0}_nplayers{1}_ppt{2}_res{3}_vals{4}".format(
        str(n_teams), str(n_players), str(players_per_team), str(reserve_price), str(player_values)
    )
    cwd = os.getcwd()
    path = cwd + "/results/" + scenario_name
    if not os.path.exists(path):
        os.makedirs(path)

    num_files = len([name for name in os.listdir(path)])
    ind = num_files // 2
    np.savez(path + "/team_returns_{}.npz".format(str(ind)), team_returns)
    np.savez(path + "/player_allocs_{}.npz".format(str(ind)), player_allocations)

    print("Done.")


