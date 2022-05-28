import numpy as np

class Auction(object):
    def __init__(self, n_teams, n_players, player_team_values, players_per_team, reserve_price, budget=None):
        self.n_teams = n_teams
        self.n_players = n_players

        self.players_per_team = players_per_team
        self.reserve_price = reserve_price

        # in the order in which they are presented.
        self.player_team_values = player_team_values
        self.budget = budget
        if self.budget is not None:
            self.state_dim = self.n_players + 1 + self.n_teams
        else:
            self.state_dim = self.n_players + 1

        # variables for maintaining auction state
        self.auction_idx = 0
        self.team_players = {i: [] for i in range(self.n_teams)}
        self.player_assignments = -1.0 * np.ones((self.n_players, ))
        # P x N (N = num teams, P = num players)
        self.value_matrix = player_team_values
        self.team_completed = np.zeros(self.n_teams)
        self.auction_history = []
        self.budgets = [budget for _ in range(self.n_teams)]

    def reset(self):
        self.auction_idx = 0
        self.team_players = {i: [] for i in range(self.n_teams)}
        self.value_matrix = np.array(self.player_team_values)
        self.team_completed = np.zeros(self.n_teams)
        self.auction_history = []
        self.budgets = [self.budget for _ in range(self.n_teams)]
        return np.stack([self.get_auction_representation(i) for i in range(self.n_teams)])

    def get_auction_representation(self, team_idx):
        players_left_to_buy = self.players_per_team - len(self.team_players[team_idx])
        if self.budget:
            reordered_budget = [self.budgets[team_idx]] + self.budgets[:team_idx] + self.budgets[team_idx+1:]
            return np.concatenate([np.array([players_left_to_buy]),
                                   np.array(reordered_budget),
                                   self.value_matrix[:, team_idx]], axis=-1).astype(np.float32)
        return np.concatenate([np.array([players_left_to_buy]),
                                   self.value_matrix[:, team_idx]], axis=-1).astype(np.float32)

    def step(self, team_bids):
        # make a copy so that any in-place edits don't impact the caller
        team_bids = team_bids.copy()
        team_bids[self.team_completed == 1.0] = -np.inf
        rewards = np.zeros(self.n_teams)

        # determine the winning team and the price
        if self.budget is not None:
            # determine who has violated the budget
            budget_violated = team_bids > np.array(self.budgets)
            # penalize teams for violating the budget
            rewards[budget_violated == 1.0] -= 100.0  # TODO: is there a smarter penalty reward we can assign?
            # teams who violate the budget are done
            self.team_completed[budget_violated == 1.0] = 1.0

            if np.all(budget_violated):
                # if everyone has violated the budget then return out.
                done = True
                states = [self.get_auction_representation(i) for i in range(self.n_teams)]
                return np.stack(states), rewards.reshape(-1, 1), self.team_completed.reshape(-1, 1).copy(), done

            # teams that violated the budget shouldn't be considered.
            team_bids[budget_violated == 1.0] = -np.inf

        argsorted_bids = np.argsort(team_bids)
        winning_team = argsorted_bids[-1]
        self.team_players[winning_team].append(self.auction_idx)
        self.player_assignments[self.auction_idx] = winning_team

        if abs(team_bids[argsorted_bids[-2]]) == np.inf:
            price = self.reserve_price
        else:
            price = team_bids[argsorted_bids[-2]]
        # team that won the player gets a payoff (reward)
        payoff = self.value_matrix[self.auction_idx, winning_team] - price

        # reward for winning team is payoff
        rewards[winning_team] = payoff
        # update the state of the auction
        self.value_matrix[self.auction_idx] = -1.0 * np.ones_like(self.value_matrix[self.auction_idx])

        if len(self.team_players[winning_team]) == self.players_per_team:
            self.team_completed[winning_team] = 1.0

        if self.budget is not None:
            self.budgets[winning_team] -= price
            if self.budgets[winning_team] <= 0:
                # if a team has used up their budget, they're done.
                self.team_completed[winning_team] = 1.0

        states = [self.get_auction_representation(i) for i in range(self.n_teams)]
        self.auction_history.append((team_bids, winning_team, price, payoff))
        done = np.all(self.team_completed)
        self.auction_idx += 1
        return np.stack(states), rewards.reshape(-1, 1), self.team_completed.reshape(-1, 1).copy(), done
