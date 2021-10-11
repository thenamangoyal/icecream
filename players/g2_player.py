import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args:
            flavor_preference (List[int]): flavor preference, most flavored flavor is first element in the list and last element is least preferred flavor
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = 0

    def get_highest_score(self,top_layer,curr_level):
        score = 0
        max_i = -1
        max_j = -1

        # Loop through every possible 2x2 square on the grid
        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                spoon_level = [curr_level[i,j],curr_level[i+1,j],curr_level[i,j+1],curr_level[i+1,j+1]]
                # if we choose this, our turn will be terminated,
                # unless it is the last step, we can use state to optimize.
                # like when 20<state<24, we can spoon 0 if the total 4 can give us greatest score.

                highest_level = max(spoon_level)
                if highest_level <= 0: # zero will get no score and -1 will get terminated, so we skip
                    continue
                curr_flavors = [top_layer[i,j],top_layer[i+1,j],top_layer[i,j+1],top_layer[i+1,j+1]]
                curr_score = 0
                for index,flavor in enumerate(curr_flavors):
                    if spoon_level[index] == highest_level:
                        # Total amount of flavors - index of this flavor (index 0 subtracts zero so player gets full points)
                        curr_score += (len(self.flavor_preference)-self.flavor_preference.index(flavor))
                if curr_score>score:
                    score=curr_score
                    max_i=i
                    max_j=j
        print(score)
        return max_i,max_j

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int, get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int], get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[str, Union[Tuple[int], int]]:
        """Request what to scoop or whom to pass in the given step of the turn. In each turn the simulator calls this serve function multiple times for each step for a single player, until the player has scooped 24 units of ice-cream or asked to pass to next player or made an invalid request. If you have scooped 24 units of ice-cream in a turn then you get one last step in that turn where you can specify to pass to a player.

        Args:
            top_layer (np.ndarray): Numpy 2d array of size (24, 15) containing flavor at each cell location
            curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from 8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
            player_idx (int): index of your player, 0-indexed
            get_flavors (Callable[[], List[int]]): method which returns a list of all possible flavors 
            get_player_count (Callable[[], int]): method which returns number of total players
            get_served (Callable[[], List[Dict[int, int]]]): method which returns a list of dictionaries corresponding to each player, each dictionary at index i tells how units of a flavor are present in the bowl of the player with index i. E.g. lets say the fourth element is {1: 0, 2: 8...} means the corresponding player with index 4 has 0 units of flavor 1 and 8 units of flavor
            get_turns_received (Callable[[], List[int]]): method which returns a list of integers corresponding to each player, each element at index i tells how many turns a player with index i has played so far.

        Returns:
            Dict[str, Union[Tuple[int],int]]: Return a dictionary specifying what action to take in the next step.
            2 possible return values
            {"action": "scoop",  "values" : (i,j)} stating to scoop the 4 cells with index (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            {"action": "pass",  "values" : i} pass to next player with index i
        """


        action = "scoop"
        values = self.get_highest_score(top_layer,curr_level)
        #     # other_player_list = list(range(0, get_player_count()))
        #     # other_player_list.remove(player_idx)
        #     # next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
        #     # action = "pass"
        #     # values = next_player
        #     # i = self.rng.integers(0, top_layer.shape[0]-1)
        #     # j = self.rng.integers(0, top_layer.shape[1]-1)
        #     action = "scoop"
        #     values = (1, 0)
        return {"action": action,  "values": values}
