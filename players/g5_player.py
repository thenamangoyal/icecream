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
        self.reverse_preference = flavor_preference[::-1]
        self.rng = rng
        self.logger = logger
        self.state = None
        self.amt_servings = 0
        self.current_turn = 0
        self.other_player_prefs = {}

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
        if get_turns_received()[player_idx] != self.current_turn:
            self.logger.info("SERVINGS G5: " + str(self.amt_servings))
            self.amt_servings = 0
            self.current_turn = self.current_turn + 1

        if self.amt_servings == 24:
            global_served = get_served()
            self.get_other_player_prefs(global_served)
            highest_score = 0
            highest_player = None
            turns_received = np.array(get_turns_received())
            valid_players = np.where(turns_received == turns_received.min())[0]
            for player in valid_players:
                total_score = self.checking_the_grid(top_layer, curr_level, self.other_player_prefs[player])
                if total_score >= highest_score:
                    highest_score = total_score
                    highest_player = player
            action = "pass"
            values = highest_player
        else:
            i, j, num_scoops = self.get_best_choice_greedy(top_layer, curr_level, 24 - self.amt_servings)
            self.amt_servings = self.amt_servings + num_scoops
            action = "scoop"
            values = (i, j)
        return {"action": action,  "values": values}

    '''
    Function to calculate preferences of other people based on the content they already have in their bowls
    Assumes that a greater number of particular flavor indicates that the person prefers that flavor
    '''
    def get_other_player_prefs(self, global_served):
        for i in range(len(global_served)):
            player_servings = sorted(global_served[i].items(), key=lambda kv: (kv[1], kv[0]))
            self.other_player_prefs[i] = [serve[0] for serve in reversed(player_servings)]

    '''
    Check the grid and assign a score to each player depending on their preference
    Returns a total of all these scores
    '''
    def checking_the_grid(self, top_layer, curr_level, prefs):
        total_score = np.zeros(top_layer.shape)
        # Each flavour is assigned the score as its index in player preferences
        num_scoops = 0
        # -1 to prevent index out of bounds
        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                num_scoops = curr_score = 0
                topmost_level = max(curr_level[i, j], curr_level[i+1, j], curr_level[i, j+1], curr_level[i+1, j+1])
                # Condition to handle empty ice cream cells
                if topmost_level == -1:
                    continue
                for x in [0, 1]:
                    for y in [0, 1]:
                        if curr_level[i+x, j+y] == topmost_level:
                            num_scoops += 1
                            curr_score += prefs.index(top_layer[i+x, j+y]) + 1
                curr_score /= num_scoops
                total_score[i, j] = curr_score

        return total_score.sum()

    '''
    Function that allows the player to make a greedy choice in their turn
    Iterates over all possible options where the spoon can be placed and calculates a score for each of the square based on the player's preferences
    Preferences are scored on the basis of their index in the flavor_preferences array
    So if there are 12 flavors, the least liked flavor would get a score 0 and the most like would get a score 11
    Returns the best possible index based on this scoring pattern
    '''
    def get_best_choice_greedy(self, top_layer, curr_level, max_num_scoops):
        best_i = 0
        best_j = 0
        # Each flavour is assigned the score as its index in player preferences
        best_score = 0
        best_num_scoops = 0
        best_layer = -1
        # -1 to prevent index out of bounds
        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                num_scoops = curr_score = 0
                topmost_level = max(curr_level[i, j], curr_level[i+1, j], curr_level[i, j+1], curr_level[i+1, j+1])
                # Condition to handle empty ice cream cells
                if topmost_level == -1:
                    continue
                for x in [0, 1]:
                    for y in [0, 1]:
                        if curr_level[i+x, j+y] == topmost_level:
                            num_scoops += 1
                            curr_score += self.reverse_preference.index(top_layer[i+x, j+y]) + 1
                curr_score /= num_scoops

                if num_scoops > max_num_scoops:
                    continue

                if curr_score > best_score:
                    best_num_scoops = num_scoops
                    best_score = curr_score
                    best_i = i
                    best_j = j
                    best_layer = topmost_level
                
                elif curr_score == best_score:
                    if topmost_level > best_layer:
                        best_num_scoops = num_scoops
                        best_score = curr_score
                        best_i = i
                        best_j = j
                        best_layer = topmost_level

        return best_i, best_j, best_num_scoops



