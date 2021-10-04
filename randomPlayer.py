import math
import numpy as np
import copy

class Player:
    def __init__(self, flavor_preference, rng, logger) -> None:
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = None
        self.initialize()

    def initialize(self):
        pass

    def get_flavor_preference(self, flavor):
        return self.flavor_preference.index(flavor) + 1
    
    def serve(self, top_layer, curr_level, player_idx, get_flavors, get_player_count, get_served, get_turns_received):
        x = self.rng.integers(0,4)
        if x < 3:
            i = self.rng.integers(0, top_layer.shape[0]-1)
            j = self.rng.integers(0, top_layer.shape[1]-1)
            action = "scoop"
            values = (i,j)
        else:
            other_player_list = list(range(0,get_player_count()))
            other_player_list.remove(player_idx)
            next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
            action = "pass"
            values = next_player
        return {"action": action,  "values" : values}
