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
        self.rng = rng
        self.logger = logger
        self.state = 0

    def get_highest_score(self,top_layer,curr_level):
        
        # Starting a new turn, we can scoop 24 units of ice cream
        if self.state == 0:
            self.state = 24
        score = 0
        max_locations = []

        # Loop through every possible 2x2 square on the grid
        for i in range(top_layer.shape[0]-1):
            for j in range(top_layer.shape[1]-1):
                spoon_level = [curr_level[i,j],curr_level[i+1,j],curr_level[i,j+1],curr_level[i+1,j+1]]

                highest_level = max(spoon_level)
                if highest_level < 0: # zero will get no score and -1 will get terminated, so we skip
                    continue
                curr_flavors = [top_layer[i,j],top_layer[i+1,j],top_layer[i,j+1],top_layer[i+1,j+1]]
                curr_score = 0
                cell_counter = 0
                for index,flavor in enumerate(curr_flavors):
                    if spoon_level[index] == highest_level:
                        cell_counter+=1
                       # Total amount of flavors - index of this flavor (index 0 subtracts zero so player gets full points)
                        curr_score += (len(self.flavor_preference)-self.flavor_preference.index(flavor))
                unit_score = curr_score / cell_counter

                # Trying to scoop more cells then we can, then we should not scoop from here
                if cell_counter > self.state:
                    continue
                if unit_score>score: #inspired by group 6 to do per unit score
                    score=unit_score
                    max_locations = [(i,j, cell_counter, highest_level)]
                elif unit_score == score:
                    max_locations.append((i,j, cell_counter, highest_level))

        if len(max_locations) == 0: # there is no scoop we can take
            self.state = 0 # means we are ready for the next turn
            return (0,0) # eventually want to make this do a choice of passing
        
        # first priority: highest_level != 0 (so we uncover something if we can)
        if len(max_locations) == 1:
            self.update_state(max_locations[0][2])
            return (max_locations[0][0], max_locations[0][1])
        
        max_locations.sort(key=lambda x: x[2], reverse=True)
        higher_level = list(filter(lambda x: x[3] != 0, max_locations)) # filter function preserves order

        if not higher_level:
            self.update_state(max_locations[0][2])
            return (max_locations[0][0], max_locations[0][1])
        else:
            self.update_state(higher_level[0][2])
            return (higher_level[0][0], higher_level[0][1]) 


    def update_state(self, units_taken):
        self.state = self.state - units_taken
        if self.state <= 0:
            self.state = -1


        
        #second priority: higher cell_counter (so we uncover more new spots)
        

        #ideas for future:
        #think about level compared to neighbors (is it beneficial to leave little 1 squares or harmful)
        #should we save units by instead prioritizing lowest cell_counter
        #consider if decision will leave us with a left over scoop we can't use
        #consider "similar" scores
        # print(score)

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

        # print(f'Player 2 state -> {self.state}')
        if self.state == -1:
            action = "pass"
            # other_player_list = list(range(0, get_player_count()))
            # other_player_list.remove(player_idx)
            # next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
            self.state = 0 # reset for next turn
            values = player_idx
        else:
            action = "scoop"
            values = self.get_highest_score(top_layer,curr_level)
        return {"action": action,  "values": values}
