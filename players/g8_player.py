import math
import numpy as np
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
        self.flavor_range = (min(self.flavor_preference), max(self.flavor_preference))
        self.rng = rng
        self.logger = logger
        self.state = None
        self.preference_estimate = None
        self.learning_rate = 0.7  # This determines how much of new information we trust
        self.curr_units_taken = 0
        self.prev_serve_dict = None
        self.decay = None
        self.player_idx = None
        self.iteration_drop = None
        self.num_turns = None
        self.preference_initialize = None
        self.alpha_weight = 0.7

        # 1 : Greedy strategy
        # 2 : 'Altruistic' greedy strategy (Check if someone else enjoys a scoop more before scooping)
        self.scooping_strategy = 2
        self.scooping_optimizations = True  # include scooping optimizations
        self.num_players = 0

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
            curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing current level at each cell location from 7 to 0, where 7 is highest level at start and -1 means no icecream left at this level
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

        self.num_players = get_player_count()
        if self.num_players == 2:
            self.scooping_strategy = 1

        # This will only happen in the second around
        if max(get_turns_received()) == 2 and not all(self.preference_initialize):
            serve_dict = get_served()
            for p_id, is_initialized in enumerate(self.preference_initialize):
                if not is_initialized:
                    cumulative = serve_dict[p_id]
                    cumulative_flavor_estimate = sorted(range(self.flavor_range[0], self.flavor_range[1] + 1),
                                                        key=lambda k: -cumulative[k])
                    self.preference_estimate[p_id] = cumulative_flavor_estimate
                    self.preference_initialize[p_id] = True

        # This will only happen in the first round
        if not self.preference_estimate:
            # Random permutation for preference estimate
            self.preference_estimate = [self.rng.permutation(self.flavor_preference).tolist() for _ in
                                        range(get_player_count())]
            self.preference_initialize = [False] * get_player_count()

            # If we are not the first group to scoop for the first time
            if sum(get_turns_received()) != 1:
                turns_num = get_turns_received()
                max_turn = max(turns_num)
                players_served = [p_id for p_id, turn in enumerate(turns_num) if p_id != player_idx and turn == max_turn]
                # We can get a more accurate estimate on the preferences on player who has been served first
                # This will overwrite the previous random permutation on estimate
                # Using player's cumulative units of ice cream to predict only happens once in the beginning
                serve_dict = get_served()
                for p_id in players_served:
                    cumulative = serve_dict[p_id]
                    cumulative_flavor_estimate = sorted(range(self.flavor_range[0], self.flavor_range[1] + 1), key=lambda k: -cumulative[k])
                    self.preference_estimate[p_id] = cumulative_flavor_estimate
                    self.preference_initialize[p_id] = True

        if not self.player_idx:
            self.player_idx = player_idx
            self.preference_estimate[self.player_idx] = self.flavor_preference

        # Create a previous serve dictionary that stores the cumulative units served for each player
        # Same structure what get_served returns
        if not self.prev_serve_dict:
            self.prev_serve_dict = [{f: 0 for f in range(self.flavor_range[0], self.flavor_range[1] + 1)} for _ in
                                    range(get_player_count())]

        if not self.num_turns:
            self.num_turns = 120 // get_player_count()

        if not self.decay:
            self.decay = self.learning_rate / self.num_turns

        if not self.iteration_drop:
            if self.num_turns <= 15:
                self.iteration_drop = 1
            else:
                self.iteration_drop = 2

        # Try to scoop if we can scoop more
        if self.curr_units_taken < 24:
            action = "scoop"
            values, units_taken, _ = self.get_max(top_layer, curr_level, self.flavor_preference, self.curr_units_taken)
            self.curr_units_taken += units_taken
            if units_taken != 0:
                return {"action": action, "values": values}

        # Pass to another player if the scoop failed or we cannot scoop anymore
        action = "pass"
        turns_num = get_turns_received()
        max_turn = max(turns_num)
        players_served = [p_id for p_id, turn in enumerate(turns_num) if p_id != player_idx and turn == max_turn]
        next_serve_dict = get_served()
        self.update_preferences(next_serve_dict)
        self.learning_rate = self.step_decay(min(turns_num))
        self.alpha_weight = self.step_decay(min(turns_num))
        players_not_served = [p_id for p_id in range(get_player_count()) if p_id not in players_served and p_id != player_idx]
        if len(players_not_served) == 0:
            players_not_served = list(range(get_player_count()))
            players_not_served.remove(player_idx)
        next_player = self.choose_player(top_layer, curr_level, players_not_served)
        values = next_player
        self.curr_units_taken = 0

        return {"action": action, "values": values}

    def update_preferences(self, new_serve_dict) -> None:
        for player_id, (prev, curr) in enumerate(zip(self.prev_serve_dict, new_serve_dict)):
            if player_id == self.player_idx:
                continue
            turn_differences = self.compute_turn_differences(prev, curr)
            # If the current player has not been served, no info update
            if sum(turn_differences) == 0:
                continue
            turn_flavor_estimate = sorted(range(1, len(turn_differences) + 1), key=lambda k: -turn_differences[k - 1])
            cumulative = new_serve_dict[player_id]
            cumulative_flavor_estimate = sorted(range(self.flavor_range[0], self.flavor_range[1] + 1), key=lambda k: -cumulative[k])
            weighted_estimate = self.compute_new_estimate(turn_flavor_estimate, cumulative_flavor_estimate, alpha=self.alpha_weight)
            self.preference_estimate[player_id] = self.compute_new_estimate(self.preference_estimate[player_id],
                                                                            weighted_estimate)

    def learning_rate_decay(self, iteration, learning_rate=None):

        '''
        Exponetially decrease the learning rate based on the number of turns
        :param learning_rate: Current learning rate
        :param iteration: int, Current turn
        :return: float, new learning rate
        '''
        if not learning_rate:
            learning_rate = self.learning_rate

        return learning_rate / (1.0 + self.decay * iteration)

    def step_decay(self, iteration, initial=0.7, drop=0.8):
        '''
        drops the learning rate by a factor every few iterations
        :param learning_rate: Current learning rate
        :param iteration: int, Current turn
        :param initial: float, initial learning rate
        :param drop: float, keep percentage
        :return: float, new learning rate
        '''

        lr = initial * math.pow(drop, math.floor((1+iteration)/self.iteration_drop))
        return lr

    def compute_turn_differences(self, prev_serve, curr_serve) -> List[int]:

        '''
        Helper method that computes the units of different flavor a player took in one turn
        :param prev_serve: previous serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player last round
        :param curr_serve: current serve dict. a dictionary that tells how many units of a flavor are present in the bowl of the player current round
        :return: a list of int (0-indexed) indicating what flavor the player took since last turn
        '''

        return [curr_serve[f] - prev_serve[f] for f in range(self.flavor_range[0], self.flavor_range[1] + 1)]

    def compute_new_estimate(self, old_estimate, new_estimate, alpha=None):

        '''
        Computer new estimate based on old estimate and our newly computed estimate from the current turn
        :param old_estimate: A list of flavors (ints) (Most preferred on the left)
        :param new_estimate: A list of flavors (ints) (Most preferred on the left)
        :param alpha: Learning rate
        :return: A list of flavors (ints) (most preferred on the left)
        '''

        if not alpha:
            alpha = self.learning_rate

        flavor_to_idx = [0] * len(old_estimate)
        for old_idx, flavor in enumerate(old_estimate):
            new_idx = new_estimate.index(flavor)
            flavor_to_idx[flavor - 1] = alpha * new_idx + (1 - alpha) * old_idx
        idx_to_flavor = sorted(range(1, len(old_estimate) + 1), key=lambda x: flavor_to_idx[x - 1])
        return idx_to_flavor

    def choose_player(self, top_layer, curr_level, players_not_served) -> int:

        '''
        Given the current top level layout of the ice cream, and suppose that a player can scoop 24 units off the top
        level, find the player who has not been served and has the best score based on our flavor estimate to pass to
        :param top_layer: Numpy 2d array of size (24, 15) containing flavor at each cell location
        :param curr_level: Numpy 2d array of size (24, 15) containing current level at each cell location from 8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
        :param players_not_served: A list of player_id (int) of those who have not been served
        :return: A player id
        '''

        best_score = -1
        next_player = self.rng.integers(0, len(players_not_served))  # Initialize a random player to pass
        max_turn = 24
        for player in players_not_served:
            curr_units_taken = curr_player_score = cur_turn = 0
            while curr_units_taken < 24 and cur_turn < max_turn:
                cur_turn += 1
                _, units_taken, score = self.get_max_greedy(top_layer, curr_level, self.preference_estimate[player],
                                                     curr_units_taken)
                curr_units_taken += units_taken
                curr_player_score += score
            if curr_player_score > best_score:
                next_player = player
                best_score = curr_player_score
        return next_player


    def get_unit_score(self, flavor, preferences):
        return len(preferences) - preferences.index(flavor)

    def get_max(self, top_layer, curr_level, preferences, curr_units_taken) -> Tuple[Tuple[int, int], int, int]:
        '''
        Greedy: find the best possible grid to scoop the ice cream based on preferences
        :param top_layer: Numpy 2d array of size (24, 15) containing flavor at each cell location
        :param curr_level: Numpy 2d array of size (24, 15) containing current level at each cell location from 8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
        :param preferences: A list flavor indicating player's preferences
        :param curr_units_taken: An int indicates how many units of ice cream the current player has taken
        :return: the best grid that maximizes player's idividual score based on player's preferences
        '''
        ret = (-1, -1)
        final_units_taken = 0
        max_score = -1
        unit_max_score, next_unit_max_score, adjacent_max_score = -1, -1, -1
        m, n = top_layer.shape
        for i in range(m - 1):
            for j in range(n - 1):
                coords = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
                max_level = max([curr_level[coord] for coord in coords])
                next_coords = [curr_level[coord] for coord in coords if max_level != curr_level[coord]]
                next_max_level = max(next_coords) if (len(next_coords)) > 0 else []

                total_score = units_taken = 0  # get score for matching maximum level
                next_total_score = next_units_taken = 0  # get score for next matching maximum level -- where next level is the next highest level (in hopes of choosing scoop that could give us a good scoop in the future)
                other_player_scores = [0]*len(self.preference_estimate)
                for coord in coords:
                    if curr_level[coord] == max_level:
                        if top_layer[coord] == -1:
                            continue
                        #cell_score = len(preferences) - preferences.index(top_layer[coord])
                        #cell_score = self.get_unit_score(top_layer[coord], preferences)
                        total_score += self.get_unit_score(top_layer[coord], preferences)
                        if self.scooping_strategy == 2:
                            for other_player_idx in range(len(self.preference_estimate)):
                                if other_player_idx != self.player_idx:
                                    other_player_scores[other_player_idx] += self.get_unit_score(top_layer[coord], self.preference_estimate[other_player_idx])
                        units_taken += 1
                    if curr_level[coord] == next_max_level and abs(max_level - curr_level[coord] + 1) <= round((24 - curr_units_taken) / 4):
                        if top_layer[coord] == -1:
                            continue
                        cell_score = len(preferences) - preferences.index(top_layer[coord])
                        next_total_score += cell_score
                        next_units_taken += 1
                if units_taken == 0:
                    continue

                # Skip this scoop if we can expect an higher overall score if another player eats it
                better_for_other_player = False
                if self.scooping_strategy == 2:
                    for other_player_score in other_player_scores:
                        if other_player_score / (len(self.preference_estimate) - 1) > total_score:
                            better_for_other_player = True
                if better_for_other_player:
                    continue
                unit_score = total_score / units_taken
                next_unit_score = next_total_score / next_units_taken if next_units_taken != 0 else 0
                adjacent_coords = [(i-1, j), (i-1, j+1), (i, j+2), (i+1, j+2), (i+2, j+1), (i+2, j), (i+1, j-1), (i, j-1)]
                num_adjacent = len(adjacent_coords)
                adjacent_scores = [self.get_unit_score(top_layer[adj_coord], preferences) if (adj_coord[0]*adj_coord[1] >= 0 and adj_coord[0] < m and adj_coord[1] < n and curr_level[adj_coord] >= max_level) else 0 for adj_coord in adjacent_coords]
                adjacent_pair_sums = [adjacent_scores[i] + adjacent_scores[i + 1] for i in range(num_adjacent-2)]
                adjacent_pair_sums.append(adjacent_scores[num_adjacent-1] + adjacent_scores[0])
                adjacent_unit_score = max(adjacent_pair_sums)
                if units_taken <= (24 - curr_units_taken):
                    if unit_score > unit_max_score:
                        max_score = total_score
                        unit_max_score, next_unit_max_score, adjacent_max_score = unit_score, next_unit_score, adjacent_unit_score
                        final_units_taken = units_taken
                        ret = (i, j)
                    if self.scooping_optimizations and unit_score == unit_max_score and \
                            ((self.num_players > 6 and (curr_units_taken + units_taken < 24 and adjacent_unit_score > adjacent_max_score) or next_unit_score > next_unit_max_score)
                             or (6 >= self.num_players > 2 and (units_taken < final_units_taken or (curr_units_taken + units_taken < 24 and adjacent_unit_score > adjacent_max_score)))):
                        max_score = total_score
                        next_unit_max_score, adjacent_max_score = next_unit_score, adjacent_unit_score
                        final_units_taken = units_taken
                        ret = (i, j)
        if final_units_taken == 0 and self.scooping_strategy == 2:
            self.scooping_strategy = 1
            ret, final_units_taken, max_score = self.get_max(top_layer, curr_level, preferences, curr_units_taken)
            self.scooping_strategy = 2
            return ret, final_units_taken, max_score
        return ret, final_units_taken, max_score

    def get_max_greedy(self, top_layer, curr_level, preferences, curr_units_taken) -> Tuple[Tuple[int, int], int, int]:
        '''
        Greedy: find the best possible grid to scoop the ice cream based on preferences
        :param top_layer: Numpy 2d array of size (24, 15) containing flavor at each cell location
        :param curr_level: Numpy 2d array of size (24, 15) containing current level at each cell location from 8 to 0, where 8 is highest level at start and 0 means no icecream left at this level
        :param preferences: A list flavor indicating player's preferences
        :param curr_units_taken: An int indicates how many units of ice cream the current player has taken
        :return: the best grid that maximizes player's idividual score based on player's preferences
        '''
        ret = (-1, -1)
        final_units_taken = 0
        max_score = -1
        unit_max_score = -1
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
                        cell_score = len(preferences) - preferences.index(top_layer[coord])
                        total_score += cell_score
                        units_taken += 1
                if units_taken == 0:
                    continue
                unit_score = total_score / units_taken
                if units_taken <= (24 - curr_units_taken) and unit_score > unit_max_score:
                    max_score = total_score
                    unit_max_score = unit_score
                    final_units_taken = units_taken
                    ret = (i, j)
        return ret, final_units_taken, max_score
