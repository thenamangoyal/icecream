import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
from operator import itemgetter


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args:
            flavor_preference (List[int]): flavor preference, most flavored flavor is first element in the list and last element is least preferred flavor
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.flavor_range = (min(self.flavor_preference), max(self.flavor_preference))
        self.rng = rng
        self.logger = logger
        self.state = None
        self.preference_estimate = None
        self.alpha = 0.3  # This determines how much of new information we trust
        self.curr_units_taken = 0
        self.prev_serve_dict = None

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
              get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]) -> Dict[
        str, Union[Tuple[int], int]]:

        """Request what to scoop or whom to pass in the given step of the turn. In each turn the simulator calls this
        serve function multiple times for each step for a single player, until the player has scooped 24 units of
        ice-cream or asked to pass to next player or made an invalid request. If you have scooped 24 units of ice-cream
        in a turn then you get one last step in that turn where you can specify to pass to a player.

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

        Returns:
            Dict[str, Union[Tuple[int],int]]: Return a dictionary specifying what action to take in the next step.
            2 possible return values
            {"action": "scoop",  "values" : (i,j)} stating to scoop the 4 cells with index (i,j), (i+1,j), (i,j+1), (i+1,j+1)
            {"action": "pass",  "values" : i} pass to next player with index i
        """
        # Create a random preference for each player; this will only happen in the very beginning
        # a list of list
        if not self.preference_estimate:
            self.preference_estimate = [self.rng.permutation(self.flavor_preference).tolist() for _ in
                                        range(get_player_count())]
        # Create a previous serve dictionary that stores the cumulative units served for each player
        # Same structure what get_served returns
        if not self.prev_serve_dict:
            self.prev_serve_dict = [{f: 0 for f in range(self.flavor_range[0], self.flavor_range[1] + 1)} for _ in
                                    range(get_player_count())]

        if self.curr_units_taken < 24:
            action = "scoop"
            values, units_taken, _ = self.get_max(top_layer, curr_level, self.flavor_preference, self.curr_units_taken)
            self.curr_units_taken += units_taken
        else:
            turns_num = get_turns_received()
            max_turn = max(turns_num)
            players_served = [p_id for p_id, turn in enumerate(turns_num) if p_id != player_idx and turn == max_turn]
            served_flavors = get_served()
            next_serve_dict = [{f: served_flavors[p][f] for f in range(self.flavor_range[0], self.flavor_range[1] + 1)} for p in
                                    range(get_player_count())]
            self.update_preferences(next_serve_dict)
            players_not_served = [p_id for p_id in range(get_player_count()) if p_id != player_idx and p_id not in players_served]
            other_player_list = list(range(0, get_player_count()))
            other_player_list.remove(player_idx)
            # next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
            next_player = self.choose_player(top_layer, curr_level, players_not_served)
            action = "pass"
            values = next_player
            self.curr_units_taken = 0

            self.prev_serve_dict = next_serve_dict
        return {"action": action, "values": values}

    def update_preferences(self, new_serve_dict) -> None:
        for player_id, (prev, curr) in enumerate(zip(self.prev_serve_dict, new_serve_dict)):
            turn_differences = self.compute_turn_differences(prev, curr)
            print("turn differences")
            print(turn_differences)
            turn_totals = self.compute_turn_totals(prev, curr)
            print("turn totals")
            print(turn_totals)
            turn_weighted_results = self.compute_turn_weighted_results(turn_differences, turn_totals)
            print("turn weighted results")
            print(turn_weighted_results)
            self.preference_estimate[player_id] = sorted(range(len(turn_weighted_results)), key=lambda k: turn_weighted_results[k], reverse=True)
            self.preference_estimate[player_id] = [val + 1 for val in self.preference_estimate[player_id]]
            print("preferences")
            print(self.preference_estimate[player_id])


    def compute_turn_differences(self, prev_serve, curr_serve) -> List[int]:

        '''
        Helper method that compute the units of different flavor a player took

        :param prev_serve:  previous serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player last round
        :param curr_serve:  current serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player current round
        :return: a list of int (0-indexed) indicating what flavor the player took since last turn
        '''

        return [curr_serve[f] - prev_serve[f] for f in range(self.flavor_range[0], self.flavor_range[1] + 1)]

    def compute_turn_totals(self, prev_serve, curr_serve) -> List[int]:

        '''
        Helper method that compute the total units of flavor a player took

        :param prev_serve:  previous serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player last round
        :param curr_serve:  current serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player current round
        :return: a list of int (0-indexed) indicating what flavors the player has taken in total
        '''

        return [curr_serve[f] + prev_serve[f] for f in range(self.flavor_range[0], self.flavor_range[1] + 1)]

    def compute_turn_weighted_results(self, differences, total) -> List[int]:

        '''
        Helper method that computes a weighted preference value for each flavor using a weight a, where value = a * total + (1-a) * differences

        :param differences:  a list of int (0-indexed) indicating what flavor the player took since last turn
        :param total:  a list of int (0-indexed) indicating what flavors the player has taken in total
        :return: a list of int (0-indexed) indicating flavor score
        '''

        a = 0.4
        return [((a * total[f]) + ((1.0 - a) * differences[f])) for f in range(self.flavor_range[0]-1, self.flavor_range[1])]

    def choose_player(self, top_layer, curr_level, players_not_served) -> int:

        best_score = -1
        next_player = self.rng.integers(0, len(players_not_served))  # Initialize a random player to pass
        for player in players_not_served:
            curr_units_taken = curr_player_score = 0
            while curr_units_taken < 24:
                _, units_taken, score = self.get_max(top_layer, curr_level, self.preference_estimate[player],
                                                     curr_units_taken)
                curr_units_taken += units_taken
                curr_player_score += score
            if curr_player_score > best_score:
                next_player = player
                best_score = curr_player_score
        return next_player

    def get_max(self, top_layer, curr_level, preferences, curr_units_taken) -> Tuple[Tuple[int, int], int, int]:

        ret = (-1, -1)
        final_units_taken = 0
        max_score = -1
        m, n = top_layer.shape
        for i in range(m - 1):
            for j in range(n - 1):
                coords = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                max_level = max([curr_level[coord] for coord in coords])
                # get score for matching maximum level
                total_score = units_taken = 0
                for coord in coords:
                    if curr_level[coord] == max_level:
                        if top_layer[coord] == -1:
                            continue
                        cell_score = len(preferences) - preferences.index(top_layer[coord]) + 1
                        total_score += cell_score
                        units_taken += 1
                if units_taken <= (24 - curr_units_taken) and total_score > max_score:
                    max_score = total_score
                    final_units_taken = units_taken
                    ret = (i, j)
        return ret, final_units_taken, max_score
