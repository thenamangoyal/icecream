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

        self.units_taken = 0
        self.prev_turn = 0

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int, get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int], get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[str, Union[Tuple[int], int]]:
        """Request what to scoop or whom to pass in the given step of the turn. In each turn the simulator calls this serve function multiple times for each step for a single player, until the player has scooped 24 units of ice-cream or asked to pass to next player or made an invalid request. If you have scooped 24 units of ice-cream in a turn then you get one last step in that turn where you can specify to pass to a player.

        Args:
            top_layer (np.ndarray): Numpy 2d array of size (24, 15) containing flavor at each cell location
            curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from 7 to -1, where 7 is highest level at start and -1 means no icecream left at this level
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

        turns_rec = get_turns_received()[player_idx]
        if self.prev_turn != turns_rec:
            self.prev_turn = turns_rec
            self.units_taken = 0

        action = "scoop"
        max_pos = (-1,-1)
        max_score = -1
        max_scoop_size = 0

        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                score, scoop_size = self.scoop_score(top_layer, curr_level, (i,j))
                if score > max_score:
                    max_score = score
                    max_pos = (i,j)
                    max_scoop_size = scoop_size
        self.units_taken += max_scoop_size

        return {"action": action,  "values": max_pos}

    def scoop_score(self, top_layer, curr_level, pos):
        """
        top_layer (np.ndarray): Numpy 2d array of size (24, 15) containing flavor at each cell location
        curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from 7 to -1, where 7 is highest level at start and -1 means no icecream left at this level
        pos (Tuple[int,int]): Position of top left of scoop

        Returns:
            score per unit in scoop
        """

        # No index out of bounds using this
        scoop_flavors = top_layer[pos[0]:pos[0]+2, pos[1]:pos[1]+2]
        scoop_levels  = curr_level[pos[0]:pos[0]+2, pos[1]:pos[1]+2]

        max_level = np.max(scoop_levels)

        score = 0
        units = 0
        
        for i,j in np.ndindex(scoop_flavors.shape):
            # unit at same level as max and the unit is actually there
            if scoop_levels[i,j] == max_level and scoop_flavors[i,j] != -1:
                # exactly how scores are computed in main.py
                score += len(self.flavor_preference) - (self.flavor_preference.index(scoop_flavors[i,j]) + 1) + 1
                units += 1
        
        if self.units_taken + units > 24: # Scoop will take more than 24
            score = 0
            units = 0
        elif units != 0:
            score /= units
        
        return score, units

    def get_scoop_size(self, curr_level, pos):
        """
        curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from 8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
        pos (Tuple[int,int]): Position of top left of scoop

        Returns:
            Number of units in this scoop
        """

        scoop_levels = curr_level[pos[0]:pos[0]+2, pos[1]:pos[1]+2]
        max_level = np.max(scoop_levels)

        scoop = scoop_levels[(scoop_levels == max_level) & (scoop_levels != -1)]
        return scoop.shape[0]
