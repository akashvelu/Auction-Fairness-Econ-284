import numpy as np

class Auction(object):
    def __init__(self, n_teams, n_players, player_team_values, players_per_team, reserve_price):
        self.n_teams = n_teams
        self.n_players = n_players

        self.players_per_team = players_per_team
        self.reserve_price = reserve_price

        # in the order in which they are presented.
        self.player_team_values = player_team_values

        self.state_dim = self.n_players + 1

        # variables for maintaining auction state
        self.auction_idx = 0
        self.team_players = {i: [] for i in range(self.n_teams)}
        # P x N (N = num teams, P = num players)
        self.value_matrix = player_team_values
        self.team_completed = np.zeros(self.n_teams)
        self.auction_history = []

    def reset(self):
        self.auction_idx = 0
        self.team_players = {i: [] for i in range(self.n_teams)}
        self.value_matrix = np.array(self.player_team_values)
        self.team_completed = np.zeros(self.n_teams)
        self.auction_history = []
        return np.stack([self.get_auction_representation(i) for i in range(self.n_teams)])

    def get_auction_representation(self, team_idx):
        players_left_to_buy = self.players_per_team - len(self.team_players[team_idx])
        return np.concatenate([np.array([players_left_to_buy]), self.value_matrix[:, team_idx]], axis=-1).astype(np.float32)

    def step(self, team_bids):
        # sort the bids, determine the winning team assign player to team and get the 2nd highest bid as price
        # assert np.min(team_bids) >= self.reserve_price
        team_bids[self.team_completed == 1.0] = -np.inf
        argsorted_bids = np.argsort(team_bids)
        winning_team = argsorted_bids[-1]
        self.team_players[winning_team].append(self.auction_idx)
        price = team_bids[argsorted_bids[-2]] if not abs(team_bids[argsorted_bids[-2]]) == np.inf else self.reserve_price

        # assign reward of 0 to all teams who didn't win the player
        payoff = self.value_matrix[self.auction_idx, winning_team] - price
        rewards = np.zeros(self.n_teams)
        rewards[winning_team] = payoff

        # update the state of the auction
        self.value_matrix[self.auction_idx] = -1.0 * np.ones_like(self.value_matrix[self.auction_idx])

        if len(self.team_players[winning_team]) == self.players_per_team:
            self.team_completed[winning_team] = 1.0

        states = [self.get_auction_representation(i) for i in range(self.n_teams)]
        self.auction_history.append((team_bids, winning_team, price, payoff))
        done = np.all(self.team_completed)
        self.auction_idx += 1
        return np.stack(states), rewards.reshape(-1, 1), self.team_completed.reshape(-1, 1).copy(), done
