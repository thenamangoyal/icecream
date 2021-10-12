import math
import random

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
                if current_level > 0 and current_level == max_level:
                    flavor_cells.append(top_layer[i + i_offset, j + j_offset])
        return flavor_cells

    def find_max_scoop(self, top_layer, curr_level, flavor_preference, max_scoop_size):
        max_scoop_loc = (0, 0)
        max_scoop_points = 0
        for i in range(len(top_layer) - 1):
            for j in range(len(top_layer[0]) - 1):
                scoop_points, scoop_size = self.calc_scoop_points(i, j, curr_level, top_layer, flavor_preference)
                if scoop_points > max_scoop_points and scoop_size <= max_scoop_size:
                    max_scoop_points = scoop_points
                    max_scoop_loc = (i, j)

        return max_scoop_loc

    def get_player_approximate_fav(self, player_count, served) -> List[int]:
        player_approximate_fav = [0 for i in range(player_count)]
        for i in range(player_count):
            player_approximate_fav[i] = max(served[i], key=served[i].get) - 1

        # adjusted to reflect 0 index
        return player_approximate_fav

    def get_top_layer_flavour_count(self, top_layer: np.ndarray) -> List[int]:
        top_layer_flavour_count = [0 for x in self.flavor_preference]

        m, n = top_layer.shape
        for i in range(m):
            for j in range(n):
                if top_layer[i][j] >= 1:
                    top_layer_flavour_count[top_layer[i][j] - 1] += 1

        # 0 indexed
        return top_layer_flavour_count

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
            '''other_player_list = list(range(0, get_player_count()))
            other_player_list.remove(player_idx)
            next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
            values = next_player'''

            not_next = 1
            #####available_players = [i for i in range(len(turns_received)) if turns_received[i]<curr_iteration]

            # calculate which flavour the top layer has the most,
            # store the value if less than 24 or store 24, since in one turn, player can only scoop 24
            top_layer_flavour_count = self.get_top_layer_flavour_count(top_layer)
            max_same_flavour = max(top_layer_flavour_count)
            max_same_flavour = min(24, max_same_flavour)

            # get topmost preference of the player = estimated as the flavour having most units in player's bowl
            player_approximate_fav = self.get_player_approximate_fav(get_player_count(), get_served())

            # check which players are available to pass to and record their index and favorite flavor
            available_player_fav = [(i, player_approximate_fav[i]) for i in range(len(player_approximate_fav)) if
                                    get_turns_received()[i] < self.curr_turn]
            self.logger.info("available player fav is {}".format(available_player_fav))
            self.logger.info("length of available player fav is {}".format(len(available_player_fav)))

             # our top half favorite flavors (floor)
            our_favs = self.flavor_preference[:(len(self.flavor_preference) // 2)]
            self.logger.info("our top half favs are {}".format(our_favs))

            # calculate a list of no conflict, available players in hope to maximize our score first before we consider other members
            no_conflict_player_fav = []
            for player, flavour in available_player_fav:
                if flavour not in our_favs:
                    no_conflict_player_fav.append((player, flavour))
            self.logger.info("no conflict player fav is {}".format(no_conflict_player_fav))
            self.logger.info("length of no conflict player fav is {}".format(len(no_conflict_player_fav)))

            # randomly select a player if while logic doesn't work
            if len(available_player_fav) > 0:
                values, flavour = choice(available_player_fav)
            else:  # we are the last member to serve in this round, pass to a random member to start the next round
                # don't want to pass to ourselves cause we should have almost exhausted our most favorite flavors on top level
                values = random.randint(0, len(get_served()))

            # take a player for the available players and check if their favorite flavour has 24 units
            # or less depending on max_same_flavour, if yes pass to that player, try to maximize family members score if no conflict
            while not_next and len(no_conflict_player_fav) > 0:
                player, flavour = no_conflict_player_fav.pop()
                if top_layer_flavour_count[flavour] >= max_same_flavour:
                    values = player

            action = "pass"
            values = values
        else:
            action = "scoop"
            values = self.find_max_scoop(top_layer, curr_level, self.flavor_preference, 24 - self.num_units_in_turn)
            self.num_units_in_turn += len(self.get_flavor_cells_from_scoop(values[0], values[1], curr_level, top_layer))

        return {"action": action, "values": values}
