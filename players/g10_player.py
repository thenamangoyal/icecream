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
        self.state = None
        self.curr_turn = 0
        self.num_scoops_in_turn = 0


    def calc_flavor_points(self, flavors_scooped, flavor_preference):
        total = 0
        for flavor_cell in flavors_scooped:
            preference_idx = flavor_preference.index(flavor_cell)
            preference_score = len(self.flavor_preference) - preference_idx
            total += preference_score
        return total


    def calc_scoop_points(self, i, j, curr_level, top_layer, flavor_preference):
        if i >= len(curr_level - 1) or j >= len(curr_level[0] - 1):
            return 0
        max_level = max(curr_level[i, j], curr_level[i, j+1], curr_level[i+1, j], curr_level[i+1, j+1])
        flavor_cells = []
        for i_offset in range(2):
            for j_offset in range(2):
                if curr_level[i + i_offset, j + j_offset] == max_level:
                    flavor_cells.append(top_layer[i + i_offset, j + j_offset])
        return self.calc_flavor_points(flavor_cells, flavor_preference)


    def find_max_scoop(self, top_layer, curr_level, flavor_preference):
        max_scoop_loc = (0, 0)
        max_scoop_points = 0
        for i in range(len(top_layer) - 1):
            for j in range(len(top_layer[0]) - 1):
                scoop_points = self.calc_scoop_points(i, j, curr_level, top_layer, flavor_preference)
                if scoop_points > max_scoop_points:
                    max_scoop_points = scoop_points
                    max_scoop_loc = (i, j)

        return max_scoop_loc




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
        if get_turns_received()[player_idx] > self.curr_turn:
            self.num_scoops_in_turn = 0
            self.curr_turn = get_turns_received()[player_idx]


        if self.num_scoops_in_turn >= 6:
            ##changes with actual computation!

            action = "pass"
            values = 0
        else:
            self.num_scoops_in_turn += 1

            action = "scoop"
            values = self.find_max_scoop(top_layer, curr_level, self.flavor_preference)

        return {"action": action,  "values": values}


