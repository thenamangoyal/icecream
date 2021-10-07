import math
import numpy as np
import copy
import logging
from typing import Dict, List

class Player:
    def __init__(self, flavor_preference: List, rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args:
            flavor_preference (List): flavor preference, least flavored flavor is first element in the list and most flavored is most preferred
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = None
    
    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int, get_flavors, get_player_count, get_served, get_turns_received) -> Dict:
        x = self.rng.random()
        if x < 0.95:
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
