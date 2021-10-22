import math
import random
import operator
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
from random import choice
from collections import defaultdict


def round_score(num):
    return round(num, 4)


def preference_distance(pref1, pref2):
    pref1_dict = {flavor: index for index, flavor in enumerate(pref1)}
    pref2_dict = {flavor: index for index, flavor in enumerate(pref2)}

    #print(pref1_dict)
    #print(pref2_dict)
    distance = 0
    for key in pref1_dict.keys():
        distance += abs(pref1_dict[key] - pref2_dict[key])
    return distance


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

    def get_all_flavors_unseen(self, top_layer, get_served):
        total_each_flavor = defaultdict(int)
        for team in get_served():
            for flavor in team:
                total_each_flavor[flavor] += team[flavor]

        for i in range(len(top_layer)):
            for j in range(len(top_layer[0])):
                if top_layer[i][j] != -1:
                    total_each_flavor[top_layer[i][j]] += 1
        return total_each_flavor

    def flavors_left(self, total_each_favor, get_flavors):
        flav_left = defaultdict(int)
        num_flavors = len(get_flavors())
        amount_of_each_flavor = (24 * 15 * 8) // num_flavors
        for flavor in total_each_favor:
            flav_left[flavor] = amount_of_each_flavor - total_each_favor[flavor]
        return flav_left

    def get_flavors_left_underneath(self, top_layer, get_served, get_flavors):
        total_each_flavor = self.get_all_flavors_unseen(top_layer, get_served)
        return self.flavors_left(total_each_flavor, get_flavors)

    def find_max_scoop(self, top_layer, curr_level, flavor_preference, max_scoop_size, divide_by_scoop_size=True):
        max_scoop_loc = (0, 0)
        max_scoop_points_per_unit = 0
        max_scoop_points = 0
        for i in range(len(top_layer) - 1):
            for j in range(len(top_layer[0]) - 1):
                scoop_points, scoop_size = self.calc_scoop_points(i, j, curr_level, top_layer, flavor_preference)
                if 0 < scoop_size <= max_scoop_size:
                    if divide_by_scoop_size:
                        scoop_points_per_unit = round_score(scoop_points / scoop_size)

                        if scoop_points_per_unit == max_scoop_points_per_unit:
                            if scoop_points > max_scoop_points:
                                max_scoop_loc = (i, j)
                                max_scoop_points = scoop_points
                        elif scoop_points_per_unit > max_scoop_points_per_unit:
                            max_scoop_loc = (i, j)
                            max_scoop_points = scoop_points
                            max_scoop_points_per_unit = scoop_points_per_unit
                    else:
                        if scoop_points > max_scoop_points:
                            max_scoop_loc = (i, j)
                            max_scoop_points = scoop_points

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

    def get_player_preferences(self, top_layer, player_count, served, turns_received, available_players, flavors_left_underneath, passTo) -> List[List[int]]:
        #asc order -> max preferred flavour at max index
        #print("serving details : ", served)
        player_preferences = [sorted(d.items(), key=operator.itemgetter(1)) for d in served]
        max_score = 0
        select = -1
        estimated_score = []
        magic_percentage = 0.07
        if passTo == 1 :
            magic_percentage = 0.2
        top_layer_flavour_count = self.get_top_layer_flavour_count(top_layer)
        #print(flavors_left_underneath)
        for i in range(len(player_preferences)):
            if i in available_players :
                score = self.get_player_score(top_layer_flavour_count,player_preferences[i])
                estimated_score.append(score)
                if score > max_score :
                    max_score = score

        #print(max_score)
        max_score *= (1-magic_percentage)
        select_players = []
        for i in range(len(estimated_score)) :
            if estimated_score[i] >= max_score :
                select_players.append(available_players[i])

        flavour_count = len(self.flavor_preference)
        same_preference_estimate = -math.inf
        if passTo == 1 :
            same_preference_estimate = math.inf
        for sp in select_players :
            sp_flavour_preferences = []
            for (flavour, count) in player_preferences[sp] :
                sp_flavour_preferences.append(flavour)
            sp_flavour_preferences = sp_flavour_preferences[::-1] #since in player_preferences most preferred flavour is last
            temp = preference_distance(self.flavor_preference, sp_flavour_preferences)
            if passTo==0 and temp > same_preference_estimate :
                select = sp
                same_preference_estimate = temp
            elif passTo==1 and temp < same_preference_estimate :
                select = sp
                same_preference_estimate = temp

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

    def return_optimal_pass(self, get_turns_received, get_player_count, get_served, player_idx, top_layer, get_flavors):
        action = "pass"
        turns_received = get_turns_received()
        curr_iteration = turns_received[player_idx]
        last_iteration = 120//get_player_count()
        passTo = 0 #different
        if get_player_count() >= 6 :
            passTo = 1 #same
        
        available_players = [i for i in range(len(turns_received)) if turns_received[i] < curr_iteration]
        flavors_left_underneath = self.get_flavors_left_underneath(top_layer, get_served, get_flavors)
        # print("pass available for : ", available_players)
        #if len(available_players) == 0 and curr_iteration==last_iteration-1:
        #    values = player_idx
        if len(available_players) == 0:
            available_players = [i for i in range(len(turns_received))]
            values = self.get_player_preferences(top_layer, get_player_count(), get_served(), turns_received,
                                                 available_players, flavors_left_underneath, passTo)
        elif len(available_players) == 1:
            values = available_players[0]
        else :
            values = self.get_player_preferences(top_layer, get_player_count(), get_served(), turns_received,
                                                 available_players, flavors_left_underneath, passTo)

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
            action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx, top_layer, get_flavors)
            # print("selected values : ", values)
            #print("top layer was :", self.get_top_layer_flavour_count(top_layer))
        else:
            action = "scoop"
            values, points = self.find_max_scoop(top_layer, curr_level, self.flavor_preference, 24 - self.num_units_in_turn, divide_by_scoop_size=True)

            #If no scoop was found, pass it
            if points == 0:
                action, values = self.return_optimal_pass(get_turns_received, get_player_count, get_served, player_idx, top_layer, get_flavors)
            else:
                self.num_units_in_turn += len(self.get_flavor_cells_from_scoop(values[0], values[1], curr_level, top_layer))

        #self.player_bowl_snapshot = get_served()
        return {"action": action, "values": values}
