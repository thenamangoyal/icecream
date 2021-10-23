import math
import numpy as np
import copy
import logging
import itertools
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
        self.nextPlayerPreference = {}
        self.previousNextPlayer = -1

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
            for i in range(0, len(curr_level)-1):
                for j in range(0, len(curr_level[0])-1):
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
                            for l in range(j, j+2):
                                if curr_level[k,l] == top:
                                    # check if over 24 and set score to 0 if thats the case
                                    scoops = scoops + 1
                                    score = score + (len(self.flavor_preference) - self.flavor_preference.index(
                                        top_layer[k][l]))
                                    if self.state + scoops >24:
                                        score = 0
                        if scoops != 0 : 
                            score = score/scoops # boundary cells causing this invalid move
                            score += top*len(self.flavor_preference)/100

                        if score > max_score:
                            max_score = score
                            min_scoops = scoops
                            best_cell = (i,j)
                        elif score == max_score and scoops >= min_scoops:
                            max_score = score
                            min_scoops = scoops
                            best_cell = (i,j)

            action = "scoop"
            self.state = self.state + min_scoops
            if min_scoops == 0:
                self.state = 24
            values = best_cell
        else: # passing to another player
            self.state = 0
            action = "pass"
            last_player_pref = self.updatePlayerPreference(self.previousNextPlayer, get_served()[self.previousNextPlayer])
            #last_player =  get_served()[self.previousNextPlayer]
            #temp = list({k: v for k, v in sorted(last_player.items(), key=lambda item: item[1])}.keys())
            #temp.reverse()
            #last_player_pref = temp
            other_player_list = list(range(0, get_player_count()))
            # get_turns_received (Callable[[], List[int]]): method which returns a list of integers corresponding to each player, each element at index i tells how many turns a player with index i has played so far.
            turns = get_turns_received()
            most_turns = max(turns)
            min_turns = min(turns)
            if most_turns!=min_turns:
                for i in range(0, len(turns)):
                    if turns[i] == most_turns:
                        other_player_list.remove(i)
            #other_player_list.remove(player_idx)
            player_to_pass = self.calculateBestScoop(top_layer, curr_level, get_flavors, other_player_list, get_served, get_turns_received, last_player_pref)
            self.nextPlayerPreference = self.estimateNextScore(player_to_pass, top_layer, curr_level, get_served)
            values = player_to_pass
            self.previousNextPlayer = player_to_pass
        return {"action": action,  "values": values}


    def estimateNextScore(self, player_to_pass: int ,top_layer: np.ndarray, curr_level: np.ndarray, get_served: Callable[[], List[Dict[int, int]]]):
        player_dict = get_served()[player_to_pass]
        permutation_dict = {}
        temp = list({k: v for k, v in sorted(player_dict.items(), key=lambda item: item[1])}.keys())
        temp.reverse()
        player_preferences = temp
        if (len(player_preferences) == 2) :
            top_flavors = player_preferences[:3]
            permutations = list(itertools.permutations(top_flavors))
        else:
            permutations = []
            permutations.append(player_preferences)
        for permutation in permutations:
            preferences = list(permutation)
            if (len(player_preferences) >2):
                for i in range(3,len(player_preferences)):
                    preferences.append(player_preferences[i])
            score, permutation_dict[tuple(preferences)] = self.calculateBestScoopForEachPlayer(top_layer, curr_level, player_preferences, player_dict )
        return permutation_dict     

    def calculateBestScoop(self, top_layer: np.ndarray, curr_level: np.ndarray, get_flavors: Callable[[], List[int]], player_list, get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]], last_player_pref) -> Dict[str, Union[Tuple[int], int]]:
        
        player_to_pass = 0
        max_players_score = 0
        for player in player_list:
            if player == self.previousNextPlayer:
                player_preferences = last_player_pref
                player_dict = get_served()[player]
            else:    
                player_dict = get_served()[player]
                temp = list({k: v for k, v in sorted(player_dict.items(), key=lambda item: item[1])}.keys())
                temp.reverse()
                player_preferences = temp
            players_score, number_of_cubes_per_flavor = self.calculateBestScoopForEachPlayer(top_layer, curr_level, player_preferences, player_dict)
            if players_score > max_players_score:
                player_to_pass = player
                max_players_score = players_score
        #print("player_to_pass " + str(player_to_pass))
        return player_to_pass       

    def calculateBestScoopForEachPlayer(self, top_layer: np.ndarray, curr_level: np.ndarray, player_preferences: List[int] , player_dict):
        number_of_cubes_per_flavor = player_dict
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
            if (i, j) not in cells_taken and curr_level[i][j] == CL and (top_layer[i][j] != -1):
                cells_taken.append((i, j))
                number_of_cubes_per_flavor[player_preferences.index(top_layer[i][j]) + 1 ] += 1
            if (i+1, j) not in cells_taken and curr_level[i+1][j]  == CL and (top_layer[i+1][j] != -1):
                cells_taken.append((i+1, j))
                number_of_cubes_per_flavor[player_preferences.index(top_layer[i+1][j]) +1 ] += 1
            if (i, j+1) not in cells_taken and curr_level[i][j+1] == CL and (top_layer[i][j+1] != -1):
                cells_taken.append((i, j+1))
                number_of_cubes_per_flavor[player_preferences.index(top_layer[i][j+1]) +1 ] += 1
            if (i+1, j+1) not in cells_taken and curr_level[i+1][j+1] == CL and (top_layer[i+1][j+1] != -1):
                cells_taken.append((i+1, j+1)) 
                number_of_cubes_per_flavor[player_preferences.index(top_layer[i+1][j+1]) +1 ] += 1
        return players_score, number_of_cubes_per_flavor

    def updatePlayerPreference(self, player, player_dict):
        best_preference = []
        min_difference= 100000000
        for preference, flavors_dict in self.nextPlayerPreference.items():
            difference = 0
            for flavor, quantity in flavors_dict.items():
                difference += abs(player_dict[flavor] - quantity)
            if difference < min_difference:
                min_difference = difference
                best_preference = preference
        return best_preference
