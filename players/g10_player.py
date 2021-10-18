import math
import random
import operator
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
from random import choice


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
        self.num_units_in_turn = 0
        #self.player_bowl_snapshot = None

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
        flavor_cells = self.get_flavor_cells_from_scoop(i, j, curr_level, top_layer)
        return self.calc_flavor_points(flavor_cells, flavor_preference), len(flavor_cells)

    def get_flavor_cells_from_scoop(self, i, j, curr_level, top_layer):
        max_level = max(curr_level[i, j], curr_level[i, j + 1], curr_level[i + 1, j], curr_level[i + 1, j + 1])
        flavor_cells = []
        for i_offset in range(2):
            for j_offset in range(2):
                current_level = curr_level[i + i_offset, j + j_offset]
                if current_level >= 0 and current_level == max_level:
                    flavor_cells.append(top_layer[i + i_offset, j + j_offset])
        return flavor_cells

    def find_max_scoop(self, top_layer, curr_level, flavor_preference, max_scoop_size, divide_by_scoop_size=True):
        max_scoop_loc = (0, 0)
        max_scoop_points = 0
        for i in range(len(top_layer) - 1):
            for j in range(len(top_layer[0]) - 1):
                scoop_points, scoop_size = self.calc_scoop_points(i, j, curr_level, top_layer, flavor_preference)
                if divide_by_scoop_size and scoop_size > 0:
                    scoop_points = scoop_points / scoop_size
                if scoop_points > max_scoop_points and scoop_size <= max_scoop_size:
                    max_scoop_points = scoop_points
                    max_scoop_loc = (i, j)

        return max_scoop_loc, max_scoop_points

    def get_player_approximate_fav(self, player_count, served) -> List[int]:
        player_approximate_fav = [0 for i in range(player_count)]
        for i in range(player_count):
            player_approximate_fav[i] = max(served[i], key=served[i].get) - 1
        # adjusted to reflect 0 index
        return player_approximate_fav

    def get_player_score(self, top_layer_flavour_count, player_preference) :
        score = 0
        flavor_preference = {}
        for i in range(len(player_preference)) :
            (key,val) = player_preference[i]
            flavor_preference[key] = i
        #print("flavour pref : ", flavor_preference)
        for i in range(len(top_layer_flavour_count)):
            score+=top_layer_flavour_count[i]*flavor_preference[i+1]
                
        #print("score = ", score)
        return score

    def get_player_preferences(self, top_layer, player_count, served, turns_received, available_players) -> List[List[int]]:
        #asc order -> max preferred flavour at max index
        #print("serving details : ", served)
        player_preferences = [sorted(d.items(), key=operator.itemgetter(1)) for d in served]
        #print("player preference : ", player_preferences)
        max_score = 0
        select = -1
        top_layer_flavour_count = self.get_top_layer_flavour_count(top_layer)
        for i in range(len(player_preferences)):
            if i in available_players : 
                #print("player ",i , " : ")
                score = self.get_player_score(top_layer_flavour_count,player_preferences[i])
                if score > max_score :
                    max_score = score
                    select = i

        # adjusted to reflect 0 index
        return select

    def get_top_layer_flavour_count(self, top_layer: np.ndarray) -> List[int]:
        top_layer_flavour_count = [0 for x in self.flavor_preference]

        m, n = top_layer.shape
        for i in range(m):
            for j in range(n):
                if top_layer[i][j] >= 1:
                    top_layer_flavour_count[top_layer[i][j] - 1] += 1

        # 0 indexed
        return top_layer_flavour_count

    def return_optimal_pass(self, get_turns_received, get_player_count, get_served, player_idx, top_layer):
        action = "pass"
        turns_received = get_turns_received()
        curr_iteration = turns_received[player_idx]
        available_players = [i for i in range(len(turns_received)) if turns_received[i] < curr_iteration]
        # print("pass available for : ", available_players)
        if len(available_players) > 1:
            values = self.get_player_preferences(top_layer, get_player_count(), get_served(), turns_received,
                                                 available_players)
        elif len(available_players) == 1:
            values = available_players[0]
        else:
            values = -1

        return action, values

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
              get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[
        str, Union[Tuple[int], int]]:
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
            self.num_units_in_turn = 0
            self.curr_turn = get_turns_received()[player_idx]

        if self.num_units_in_turn >= 24:
            action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx, top_layer)
            # print("selected values : ", values)
            #print("top layer was :", self.get_top_layer_flavour_count(top_layer))
        else:
            action = "scoop"
            values, points = self.find_max_scoop(top_layer, curr_level, self.flavor_preference, 24 - self.num_units_in_turn, divide_by_scoop_size=True)

            #If no scoop was found, pass it
            if points == 0:
                action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx, top_layer)
            else:
                self.num_units_in_turn += len(self.get_flavor_cells_from_scoop(values[0], values[1], curr_level, top_layer))

        #self.player_bowl_snapshot = get_served()
        return {"action": action, "values": values}
