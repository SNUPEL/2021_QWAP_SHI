import random
import numpy as np


class Heuristic:
    def __init__(self, num_of_ships, num_of_quays):
        self.num_of_ships = num_of_ships
        self.num_of_quays = num_of_quays

    def act(self, state):
        priority_idx_ship = np.full(self.num_of_ships, -1.0)
        for ship_id, value in state.items():
            priority_idx_ship[ship_id] = value[0]
        ship_idx = np.random.choice(np.where(priority_idx_ship == np.max(priority_idx_ship))[0])

        priority_idx_quay_wall = np.full(self.num_of_quays, -1.0)
        for quay_wall_id, value in state[ship_idx][1].items():
            priority_idx_quay_wall[quay_wall_id] = value
        quay_idx = np.random.choice(np.where(priority_idx_quay_wall == np.max(priority_idx_quay_wall))[0])

        return ship_idx * self.num_of_quays + quay_idx