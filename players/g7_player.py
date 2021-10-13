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
        self.flavor_points = {}
        self.average = (max(flavor_preference) + min(flavor_preference)) / 2
        self.discount = 0.9
        self.level_coef = 1

        flavor_preference_len = len(flavor_preference)

        for i in range(len(flavor_preference)):
            self.flavor_points[flavor_preference[i]] = flavor_preference_len - i

        self.flavor_points[-1] = self.average


    def calculate_score_scoop(self, scoop):
        """
        Calculate of scoop based on flavor_points
        """
        total = 0
        num_scooped = 0
        coef = self.level_coef

        for level in scoop:
            if len(level) > 0:
                for f in level:
                    total += self.flavor_points[f] * coef
                    num_scooped += 1
                coef *= self.discount
        
        return total / num_scooped if num_scooped > 0 else total

    def get_scoop(self,i,j,top_layer,curr_level):
        """
        Gets a 2x2 scoop from (i,j) to (i+1,j+1), from maximum level to minimum level
        returns list of list (e.g [[flavors on level 8], [flavors on level 7], ...])
        if there are cells that are unknown, replaces them with average value
        """
        levels = []

        scoop = curr_level[i:i+2][:,[j,j+2]]
        max_level = np.amax(scoop)
        min_level = np.amin(scoop)

        for level in range(max_level,min_level-1,-1):
            l = []
            for x in range(2):
                for y in range(2):
                    if curr_level[i+x][j+y] == level:
                        l.append(top_layer[i+x][j+y])
                    elif curr_level[i+x][j+y] > level:
                        l.append(-1)
            levels.append(l)
            
        return levels


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

        max_scoop_i, max_scoop_j = -1, -1
        max_scoop_point = -1

        for i in range(len(top_layer) - 2):
            for j in range(len(top_layer[i]) - 2):
                scoop = self.get_scoop(i,j,top_layer,curr_level)
                scoop_point = self.calculate_score_scoop(scoop)

                if scoop_point > max_scoop_point:
                    max_scoop_point = scoop_point
                    max_scoop_i, max_scoop_j = i, j

        return {"action": "scoop",  "values": (max_scoop_i, max_scoop_j)}

