import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
import heapq
from collections import defaultdict


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
        self.discount = 0.9
        self.level_coef = 1

        flavor_preference_len = len(flavor_preference)

        for i in range(len(flavor_preference)):
            self.flavor_points[flavor_preference[i]] = flavor_preference_len - i

        self.flavor_points[-1] = 0

        self.move_scores = {}
        self.round = 1
        self.priority_moves = [] # to store cumulative moves

        self.total_cells = 2880
        self.distribution = self.get_init_distribution()

        self.remaining_scoops = 24

    def get_init_distribution(self):
        flavor_preference_len = len(self.flavor_preference)
        return [self.total_cells / flavor_preference_len] * flavor_preference_len

    def get_distribution(self, served, top_layer):
        # index of distribution corresopnds to flavor with value i + 1
        self.distribution = self.get_init_distribution()
        for row in top_layer:
            for val in row:
                if val > 0:
                    self.distribution[val - 1] -= 1
        for player in served:
            for flavor, consumed in player.items():
                self.distribution[flavor - 1] -= consumed

    def update_hidden_cell_expectation(self, served, top_layer):
        self.get_distribution(served, top_layer)
        total = sum(self.distribution)
        if total == 0:
            self.flavor_points[-2] = 0
        else:
            ans = 0
            for i, v in enumerate(self.distribution):
                ans += self.flavor_points[i + 1] * max(v, 0) / total
            self.flavor_points[-2] = ans


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

        scoop = [[curr_level[i][j],curr_level[i][j+1]],[curr_level[i+1][j],curr_level[i+1][j+1]]]
        max_level = np.amax(scoop)
        min_level = max(np.amin(scoop),1)

        for level in range(max_level,min_level-1,-1):
            l = []
            for x in range(2):
                for y in range(2):
                    if curr_level[i+x][j+y] == level:
                        l.append(top_layer[i+x][j+y])
                    elif curr_level[i+x][j+y] > level:
                        l.append(-2)
            levels.append(l)
            
        return levels

    def calc_scr_flavor_pref(self, bowl):
        total = 0
        for k,v in bowl.items():
            total += self.flavor_points[k] * v
        return total

    def total_scoop_in_bowl(self, player_idx, served):
        return sum(served[player_idx].values())

    def updated_score(self, i, j, top_layer, curr_level):
        """
        Updated scoop + score function
        """

        scoop = [[curr_level[i][j], curr_level[i][j + 1]], [curr_level[i + 1][j], curr_level[i + 1][j + 1]]]
        max_level = np.amax(scoop)
        min_level = max(np.amin(scoop), 0)  # changed 1 to zero as -1 is no icecream
        cumulative_score = 0
        num_scooped = 0
        num_moves = 0
        prev_num_unseen_cells = 0
        num_unseen_cells = 0

        for level in range(max_level, min_level - 1, -1):
            for x in range(2):
                for y in range(2):
                    if curr_level[i + x][j + y] == level:
                        cumulative_score += self.flavor_points[top_layer[i + x][j + y]]
                        num_scooped += 1
                        num_unseen_cells += 1

            num_moves += 1
            cumulative_score += prev_num_unseen_cells * self.flavor_points[-2]
            num_scooped += prev_num_unseen_cells
            score = (cumulative_score / num_scooped) if num_scooped > 0 else 0
            self.move_scores[-score] = (num_moves, num_scooped, i, j)
            prev_num_unseen_cells = num_unseen_cells

    def calculate_last_2(self):
        scoops_scores = defaultdict(int)
        scoop_info = {}

        for score in self.move_scores:
            max_num_moves, max_scoop_cells, max_scoop_i, max_scoop_j = self.move_scores[score]

            for s in range(1,8):
                if s in scoop_info and s + max_scoop_cells <= self.remaining_scoops:
                    other_s_info = scoop_info[s]
                    if scoops_scores[s + max_scoop_cells] > scoops_scores[s] + score:
                        scoops_scores[s + max_scoop_cells] = scoops_scores[s] * other_s_info[1] + score * max_scoop_cells
                        if scoops_scores[s] * other_s_info[1] < score * max_scoop_cells:
                            scoop_info[s + max_scoop_cells] = other_s_info
                        else:
                            scoop_info[s + max_scoop_cells] = self.move_scores[score]
            
            if max_scoop_cells <= self.remaining_scoops:
                if max_scoop_cells not in scoop_info or scoops_scores[max_scoop_cells] * scoop_info[max_scoop_cells][1] > score * max_scoop_cells:
                    scoops_scores[max_scoop_cells] = score * max_scoop_cells
                    scoop_info[max_scoop_cells] = self.move_scores[score]
        
        for key,value in scoops_scores.items():
            self.move_scores[value] = scoop_info[key]


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

        self.update_hidden_cell_expectation(get_served(), top_layer)

        remaining_scoops = self.remaining_scoops
        if len(self.priority_moves) > 0:
            max_scoop_i, max_scoop_j = self.priority_moves.pop(0)
            max_scoop_cells = 0
            scoop = [[curr_level[max_scoop_i][max_scoop_j], curr_level[max_scoop_i][max_scoop_j + 1]], [curr_level[max_scoop_i + 1][max_scoop_j], curr_level[max_scoop_i + 1][max_scoop_j + 1]]]
            max_level = np.amax(scoop)

            for x in range(2):
                for y in range(2):
                    if curr_level[max_scoop_i+x][max_scoop_j+y] == max_level:
                        max_scoop_cells += 1
            
        else:
            for i in range(len(top_layer) - 1):
                for j in range(len(top_layer[i]) - 1):
                    self.updated_score(i, j, top_layer, curr_level)

            scores = list(self.move_scores.keys())
            if len(scores) == 0:
                time_to_pass = True
            else:
                heapq.heapify(scores)
                max_score = heapq.heappop(scores)
                max_num_moves, max_scoop_cells, max_scoop_i, max_scoop_j = self.move_scores[max_score]
                while max_scoop_cells > remaining_scoops and len(scores) > 0:
                    max_score = heapq.heappop(scores)
                    max_num_moves, max_scoop_cells, max_scoop_i, max_scoop_j = self.move_scores[max_score]

                if max_num_moves > 1:
                    max_scoop_cells = 0
                    scoop = [[curr_level[max_scoop_i][max_scoop_j], curr_level[max_scoop_i][max_scoop_j + 1]], [curr_level[max_scoop_i + 1][max_scoop_j], curr_level[max_scoop_i + 1][max_scoop_j + 1]]]
                    max_level = np.amax(scoop)

                    for x in range(2):
                        for y in range(2):
                            if curr_level[max_scoop_i+x][max_scoop_j+y] == max_level:
                                max_scoop_cells += 1
                    
                    for _ in range(max_num_moves-1):
                        self.priority_moves.append((max_scoop_i, max_scoop_j))

        time_to_pass = (remaining_scoops == 0)
        self.move_scores = {}

        if time_to_pass or max_scoop_cells > remaining_scoops:
            points = []
            served = get_served()
            turns = get_turns_received()
            for idx in range(get_player_count()):
                point = self.calc_scr_flavor_pref(served[idx])
                points.append((point,turns[idx],idx))
            
            sorted_points = sorted(points, key=lambda x: (x[1], x[0]))
            next_player = sorted_points[0][2]
            self.round += 1
            self.priority_moves = []
            self.remaining_scoops = 24
            return {"action": "pass", "values": next_player}

        self.remaining_scoops -= max_scoop_cells
        return {"action": "scoop",  "values": (max_scoop_i, max_scoop_j)}

