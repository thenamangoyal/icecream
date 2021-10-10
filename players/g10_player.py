import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
from random import choice


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        """Initialise the player with given preference.

        Args:
            flavor_preference (List[int]): flavor preference, least flavored flavor is first element in the list and most flavored is most preferred
            rng (np.random.Generator): numpy random number generator, use this for same player behvior across run
            logger (logging.Logger): logger use this like logger.info("message")
        """
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = None

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

        #state to keep track of how many times serve was called
        if self.state==None :
            self.state = [1]
        else :
            self.state[0]+=1

        #since we keep scooping for 6 times
        if self.state[0]%7 != 0:
            '''i = self.rng.integers(0, top_layer.shape[0]-1)
            j = self.rng.integers(0, top_layer.shape[1]-1)'''
            temp = self.state[0]//7
            values = (2+temp, 3+temp)
            action = "scoop"
            
            
        else:
            '''other_player_list = list(range(0, get_player_count()))
            other_player_list.remove(player_idx)
            next_player = other_player_list[self.rng.integers(0, len(other_player_list))]
            values = next_player'''

            action = "pass"

            #calculate the current iteration based on our player index, since our player is called latest
            turns_received = get_turns_received()
            curr_iteration = turns_received[player_idx]
            not_next = 1

            #####available_players = [i for i in range(len(turns_received)) if turns_received[i]<curr_iteration]  

            #calculate the max amount of flavour visible on the top layer, store the value if less than 24 or store 24, since in one turn, player can only scoop 24
            top_layer_flavour_count = self.get_top_layer_flavour_count(top_layer)
            max_same_flavour = max(top_layer_flavour_count)
            max_same_flavour = min(24, max_same_flavour)

            #get topmost preference of the player = estimated as the flavour having most units in player's bowl
            player_approximate_fav = self.get_player_approximate_fav(get_player_count(), get_served())

            #use curr_iteration to check which players are available to pass
            available_player_fav = [(i, player_approximate_fav[i]) for i in range(len(player_approximate_fav)) if turns_received[i]<curr_iteration]

            #randomly select a player if while logic doesn't work, given our player is not last in current iteration
            if len(available_player_fav) > 0 :
                values, flavour = choice(available_player_fav)
            else : #pass to ourself in the next iteration
                values = player_idx

            #take a player for the available players and check if his flavour preference has 24 units or less depending on max_same_flavour, if yes pass to that player
            while not_next and len(available_player_fav)>0 :
                player, flavour = available_player_fav.pop()
                if top_layer_flavour_count[flavour] >= max_same_flavour :
                    values = player
                    not_next = 0
            
        return {"action": action,  "values": values}

    def get_player_approximate_fav(self, player_count, served) -> List[int] :
        player_approximate_fav = [0 for i in range(player_count)]
        for i in range(player_count) :
            player_approximate_fav[i] = max(served[i],key=served[i].get)-1

        #adjusted to refect 0 index
        return player_approximate_fav

    def get_top_layer_flavour_count(self, top_layer:np.ndarray) -> List[int] :
        top_layer_flavour_count = [0 for x in self.flavor_preference]

        m,n = top_layer.shape
        for i in range(m) :
            for j in range(n) :
                if top_layer[i][j] >= 1 :
                    top_layer_flavour_count[top_layer[i][j]-1]+=1

        #0 indexed
        return top_layer_flavour_count
        

