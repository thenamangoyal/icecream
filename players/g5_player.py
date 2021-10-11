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
        self.reversed = flavor_preference[::-1]
        self.rng = rng
        self.logger = logger
        self.state = None

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
        # x = self.rng.random()
        # if x < 0.95:
        #     i = self.rng.integers(0, top_layer.shape[0]-1)
        #     j = self.rng.integers(0, top_layer.shape[1]-1)
        #     action = "scoop"
        #     values = (i, j)
        # else:
        # other_player_list = list(range(0, get_player_count()))
        # other_player_list.remove(player_idx)
        # next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
        # action = "pass"
        # values = next_player
        i, j = self.get_best_choice_greedy(top_layer, curr_level)
        action = "scoop"
        values = (i,j)
        return {"action": action,  "values": values}

    '''
    Function that allows the player to make a greedy choice in their turn
    Iterates over all possible options where the spoon can be placed and calculates a score for each of the square based
     on the player's preferences
    Preferences are scored on the basis of their index in the flavor_preferences array
    So if there are 12 flavors, the least liked flavor would get a score 0 and the most like would get a score 11
    Returns the best possible index based on this scoring pattern
    '''
    def get_best_choice_greedy(self, top_layer, curr_level):
        best_i = 0
        best_j = 0
        # Each flavour is assigned the score as its index in player preferences
        best_score = 0
        # -1 to prevent index out of bounds
        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                topmost_level = max(curr_level[i, j], curr_level[i+1, j], curr_level[i, j+1], curr_level[i+1, j+1])
                # Condition to handle empty ice cream cells
                if topmost_level == -1:
                    continue
                curr_score = 0
                for x in [0, 1]:
                    for y in [0, 1]:
                        if curr_level[i+x, j+y] == topmost_level:
                            curr_score += self.reversed.index(top_layer[i+x, j+y]) + 1

                if curr_score > best_score:
                    best_score = curr_score
                    best_i = i
                    best_j = j

        # Check scores of the last column, except last cell
        j = top_layer.shape[1]-1
        for i in range(top_layer.shape[0]-1):
            topmost_level = max(curr_level[i, j], curr_level[i + 1, j])
            if topmost_level == -1:
                continue
            curr_score = 0
            if curr_level[i, j] == topmost_level:
                curr_score += self.reversed.index(top_layer[i, j]) + 1
            if curr_level[i + 1, j] == topmost_level:
                curr_score += self.reversed.index(top_layer[i + 1, j]) + 1
            if curr_score > best_score:
                best_score = curr_score
                best_i = i
                best_j = j

        # Check scores of the last row, except last cell
        i = top_layer.shape[0] - 1
        for j in range(top_layer.shape[1] - 1):
            topmost_level = max(curr_level[i, j], curr_level[i, j + 1])
            if topmost_level == -1:
                continue
            curr_score = 0
            if curr_level[i, j] == topmost_level:
                curr_score += self.reversed.index(top_layer[i, j]) + 1
            if curr_level[i, j + 1] == topmost_level:
                curr_score += self.reversed.index(top_layer[i, j + 1]) + 1
            if curr_score > best_score:
                best_score = curr_score
                best_i = i
                best_j = j

        return best_i, best_j



