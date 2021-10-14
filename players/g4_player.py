import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args:
            flavor_preference (List[int]): flavor preference, most flavored
                flavor is first element in the list and last element is least
                preferred flavor

            rng (np.random.Generator): numpy random number
                generator, use this for same player behavior across run

            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = {
            'current_served': None,
            # Number of scoops served to ourselves in the current turn
            'current_turn_served': 0
        }

    @staticmethod
    def valid_scoop(curr_level, x, y):
        """Helper function: returns whether a scoop at an index x,y is valid or not"""
        d = curr_level[x,y]
        if curr_level[x+1,y] <= d and curr_level[x,y+1] <= d and curr_level[x+1,y+1] <= d:
            return True
        return False

    @staticmethod
    def scoop_value(flavor_preference, top_layer, curr_level, x, y):
        """Helper function: returns the value the player gets for a scoop at index x,y"""
        d = curr_level[x, y]
        try:
            if d > 0:
                units = 1
                flav_total = len(flavor_preference) - flavor_preference.index(top_layer[x,y]) + 1
                if curr_level[x+1, y] == d:
                    flav_total += len(flavor_preference) - flavor_preference.index(top_layer[x+1,y]) + 1
                    units += 1
                if curr_level[x, y+1] == d:
                    flav_total += len(flavor_preference) - flavor_preference.index(top_layer[x,y+1]) + 1
                    units += 1
                if curr_level[x+1, y+1] == d:
                    flav_total += len(flavor_preference) - flavor_preference.index(top_layer[x+1,y+1]) + 1
                    units += 1
                return (flav_total, (x, y), units)
        except ValueError:
            # No knowledge of player's preference for some value
            pass
        return (0, (x,y), 0)

    @staticmethod
    def score_available_scoops(flavor_preference, top_layer, curr_level):
        p_queue = []
        # Subtract one from length since 2x2 "spoon" must remain in container
        for x in range(0, top_layer.shape[0]-1):
            for y in range(0, top_layer.shape[1]-1):
                if Player.valid_scoop(curr_level, x, y):
                    p_queue.append(Player.scoop_value(flavor_preference, top_layer, curr_level, x, y))
        # TODO (etm): If we care, we can use an actual heap / priority queue
        p_queue.sort()
        return p_queue

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
          get_flavors: Callable[[], List[int]],
          get_player_count: Callable[[], int], get_served: Callable[[], List[Dict[int, int]]],
          get_turns_received: Callable[[], List[int]]) -> Dict[str, Union[Tuple[int], int]]:
        """Request what to scoop or whom to pass in the given step of the turn.
        In each turn the simulator calls this serve function multiple times for
        each step for a single player, until the player has scooped 24 units of
        ice-cream or asked to pass to next player or made an invalid request.
        If you have scooped 24 units of ice-cream in a turn then you get one
        last step in that turn where you can specify to pass to a player.

        Args:
            top_layer (np.ndarray): Numpy 2d array of size (24, 15) containing
                flavor at each cell location
            
            curr_level (np.ndarray): Numpy 2d array of size (24, 15) containing
                current level at each cell location from 8 to 0, where 8 is highest
                level at start and 0 means no icecream left at this level
            
            player_idx (int): index of your player, 0-indexed

            get_flavors (Callable[[], List[int]]): method which returns a list of all
                possible flavors

            get_player_count (Callable[[], int]): method which returns number of total players

            get_served (Callable[[], List[Dict[int, int]]]): method which returns a list of dictionaries
                corresponding to each player, each dictionary at index i tells how
                units of a flavor are present in the bowl of the player with index
                i. E.g. lets say the fourth element is {1: 0, 2: 8...} means the
                corresponding player with index 4 has 0 units of flavor 1 and 8
                units of flavor

            get_turns_received (Callable[[], List[int]]):
                method which returns a list of integers corresponding to each
                player, each element at index i tells how many turns a player with
                index i has played so far.

        Returns:
            Dict[str, Union[Tuple[int],int]]: Return a dictionary specifying
            what action to take in the next step.

            2 possible return values
            {"action": "scoop",  "values" : (i,j)}
            stating to scoop the 4 cells with index (i,j), (i+1,j), (i,j+1), (i+1,j+1)

            {"action": "pass", "values" : i} pass to next player with index i
        """
        # build priority queue
        p_queue = self.score_available_scoops(self.flavor_preference, top_layer, curr_level)
        # if there is still more ice-cream to take, make a scoop
        if self.state['current_turn_served'] < 24:
            action = "scoop"
            value, (x, y), units = p_queue.pop()
            self.state['current_turn_served'] += units
            values = (x, y)
        else:
            self.state['current_turn_served'] = 0
            next_player = Player.best_player_to_pass_to(player_idx, get_player_count(), top_layer, curr_level, get_served, get_flavors, get_turns_received())
            action = "pass"
            values = next_player

        # get knowledge of other players
        # pref_ranking = self.guess_player_pref_from_bowl(0, get_served(), get_flavors())  # [0] - favorite
        # added_flavors = self.diff_served(get_served(), self.state['current_served'])

        # update current served
        self.state['current_served'] = get_served()

        return {"action": action, "values": values}

    @staticmethod
    def best_player_to_pass_to(self_ix, n_players, top_layer, curr_level, get_served, get_flavors, turns_received):
        # list of all players except ourself and those who have had more turns than the current min
        players = []
        min_received = np.amin(turns_received)
        for i in range(n_players):
            if i != self_ix and turns_received[i] == min_received:
                players.append(i)

        max_player = players[0]
        max_score = 0
        for player in players:
            p_score = 0
            player_pref = Player.guess_player_pref_from_bowl(player, get_served(), get_flavors())
            # TODO (etm):
            #   This is a crude approximation since some scoops will contain chunks of
            #   other scoops. We need a better way to update the game state
            p_queue = Player.score_available_scoops(player_pref, top_layer, curr_level)
            for _ in range(24):
                if len(p_queue) == 0:
                    break
                score, _, _ = p_queue.pop()
                p_score += score
            if p_score > max_score:
                max_score = p_score
                max_player = player

        return max_player

    @staticmethod
    def guess_player_pref_from_bowl(player, current_served, flavors):
        """
        Sort the contents of player's current_served bowl and generate a preference ranking accordingly.
        :param player: target player.
        :param current_served: player's bowl.
        :param flavors: all flavors.
        :return: a list of flavor rankings. Preferred flavors are ranked toward the front.
        """
        bowl = current_served[player]
        flavors_in_bowl, amounts_in_bowl = [], []
        for key in bowl.keys():
            if bowl[key] > 0:
                flavors_in_bowl.append(key)
                amounts_in_bowl.append(bowl[key])
        amount_indices = np.argsort(amounts_in_bowl)[::-1]
        pref_ranking = [flavors_in_bowl[i] for i in amount_indices]

        # TODO what if some flavor(s) didn't appear in the bowl? Do we pad the ranking list at the end?
        pref_ranking += [0] * (len(flavors) - len(pref_ranking))

        return pref_ranking

    def guess_player_dislikes_from_choice(self, player, added_flavors, top_layer_before_serving):
        """
        Guess the flavors that player probably dislikes based on the top_layer situation before they serve themselves,
        and what flavors they end up choosing. Can only be used when we pass the container directly to player.
        :param player: the player to be guessed on.
        :param added_flavors: output of diff_served().
        :param top_layer_before_serving: visible flavors on the top layer when we pass the container to player.
        :return: a list of flavors that the player probably doesn't like. Currently the strategy is to choose flavors
                 that are on top_layer but the player doesn't end up choosing.
        TODO: this function needs to be called one turn after we pass the container to player, which is impossible.
              Currently it's not being used but we may use the idea later.
        """
        if len(added_flavors[player]) == 0:
            print("WARNING: No added flavor info for this player!")
            return None
        dislikes = []
        top_layer_summed = self.sum_top_layer(top_layer_before_serving)
        top_layer_flavors = top_layer_summed.keys()
        for flavor in top_layer_flavors:
            if int(flavor) not in added_flavors[player]:
                dislikes.append(flavor)

        return dislikes

    @staticmethod
    def is_valid_next_player(turns_received, next_player):
        """
        Checks if next_player is a valid choice for passing the container to for this turn.
        :param turns_received: output of get_turns_received().
        :param next_player: player in question.
        :return: True if next_player is a valid choice.
        """
        if np.amin(turns_received) < turns_received[next_player]:
            return False
        return True

    @staticmethod
    def diff_served(new_served, current_served=None):
        """
        Compare new_served against current_served and output changes in the container.
        :param new_served: new container situation.
        :param current_served: current container situation. None if new_served is from the first turn.
        :return: changed_flavors[player_i] contains a list of flavors that player_i has newly added to
                 their bowl since current_served.
        TODO: Take the amount into consideration?
        """
        changed_flavors = []

        for p_id in range(len(new_served)):
            changed_flavors.append([])
            for flavor in new_served[p_id].keys():
                if current_served:
                    if new_served[p_id][flavor] > current_served[p_id][flavor]:
                        changed_flavors[p_id].append(flavor)
                else:
                    if new_served[p_id][flavor] > 0:
                        changed_flavors[p_id].append(flavor)

        return changed_flavors

    @staticmethod
    def sum_top_layer(top_layer):
        """
        Summarize top_layer information by flavor.
        :param top_layer: visible flavors.
        :return: top_layer_summed['flavor_i'] is the amount of flavor_i currently visible on top_layer.
        TODO: come up with other ways to convert the top layer flavor info into useful information.
        """
        top_layer_summed = {}
        for i in range(np.shape(top_layer)[0]):
            for j in range(np.shape(top_layer)[1]):
                if str(top_layer[i][j]) not in top_layer_summed:
                    top_layer_summed[str(top_layer[i][j])] = 0
                top_layer_summed[str(top_layer[i][j])] += 1

        return top_layer_summed

