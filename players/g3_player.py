import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
from scipy import stats


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
        #x = self.rng.random()
        if self.state < 24:
            max_score = 0
            min_scoops = 4
            best_cell = (0,0)
            for i in range(0, len(curr_level) - 1):
                for j in range(0, len(curr_level[0]) - 1):
                    # first find the cells of ice cream on the top level we can take
                    options = []
                    options.append(curr_level[i][j])
                    options.append(curr_level[i+1][j])
                    options.append(curr_level[i][j+1])
                    options.append(curr_level[i+1][j+1])
                    top = max(options)
                    if top > -1:
                        # find the score for this layer
                        score = 0
                        scoops = 0
                        for k in range(i, i+2):
                            for l in range(j, j +2):
                                if curr_level[k,l] == top:
                                    scoops = scoops + 1
                                    if self.state + scoops <=24:
                                        score = score + (len(self.flavor_preference) - self.flavor_preference.index(top_layer[k][l]))

                        if score > max_score:
                            max_score = score
                            min_scoops = scoops
                            best_cell = (i,j)
                        elif score == max_score:
                            if scoops < min_scoops:
                                max_score = score
                                min_scoops = scoops
                                best_cell = (i,j)

            action = "scoop"
            self.state = self.state + min_scoops
            if min_scoops == 0:
                self.state = 24
            values = best_cell
        else: # passing to another player
            other_player_list = list(range(0, get_player_count()))
            other_player_list.remove(player_idx)
            player_to_pass = Player.calculateBestScoop(self, top_layer, curr_level, get_flavors, other_player_list, get_served, get_turns_received)
                
            action = "pass"
            values = player_to_pass
            self.state = 0
        return {"action": action,  "values": values}

    def calculateBestScoop(self, top_layer: np.ndarray, curr_level: np.ndarray, get_flavors: Callable[[], List[int]], player_list, get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[str, Union[Tuple[int], int]]:
        player_to_pass = 0
        max_players_score = 0
        for player in player_list:
            player_dict = get_served()[player]
            temp = list({k: v for k, v in sorted(player_dict.items(), key=lambda item: item[1])}.keys())
            temp.reverse()
            player_preferences = temp
            players_score = 0
            counter = 1
            cells_taken = []
            while len(cells_taken) < 24 and counter != len(cells_taken):
                counter = len(cells_taken)
                max_score = 0
                min_scoops = 4
                best_cell = (0,0)
                for i in range(0, len(curr_level) - 1):
                    for j in range(0, len(curr_level[0]) - 1):
                        # first find the cells of ice cream on the top level we can take
                        options = []
                        options.append(curr_level[i][j])
                        options.append(curr_level[i+1][j])
                        options.append(curr_level[i][j+1])
                        options.append(curr_level[i+1][j+1])
                        top = max(options)
                        if top > 0:
                            # find the score for this layer
                            score = 0
                            scoops = 0
                            for k in range(i, i+2):
                                for l in range(j, j +2):
                                    if curr_level[k,l] == top:
                                        if (k, l) not in cells_taken:
                                            score = score + (len(player_preferences) - player_preferences.index(top_layer[k][l]))
                                            scoops = scoops + 1
                                                
                            if score > max_score:
                                max_score = score
                                min_scoops = scoops
                                best_cell = (i,j)
                            elif score == max_score:
                                if scoops < min_scoops:
                                    max_score = score
                                    min_scoops = scoops
                                    best_cell = (i,j)
                players_score += max_score
                (i, j) = best_cell
                CL = max(curr_level[i][j], curr_level[i+1][j], curr_level[i][j+1], curr_level[i+1][j+1])
                if (i, j) not in cells_taken and curr_level[i][j] == CL:
                    cells_taken.append((i, j))
                if (i+1, j) not in cells_taken and curr_level[i+1][j] == CL:
                    cells_taken.append((i+1, j))
                if (i, j+1) not in cells_taken and curr_level[i][j+1] == CL:
                    cells_taken.append((i, j+1))
                if (i+1, j+1) not in cells_taken and curr_level[i+1][j+1] == CL:
                    cells_taken.append((i+1, j+1))
            if players_score > max_players_score:
                player_to_pass = player
                max_players_score = players_score
        
        return player_to_pass        
