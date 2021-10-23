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
        #self.turns = 0

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
        #max_locations.sort(key=lambda x: x[2])
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

    def get_unserved_players(self, players, player_idx, get_turns_recieved):
        min_turns = float("inf")
        passing_options = []
        turns = get_turns_recieved()
        for i in range(len(players)):
            if i == player_idx: #we aren't an option for ourselves here
                continue
            t = turns[i]
            if t == min_turns:
                passing_options.append(i)
            elif t < min_turns:
                min_turns = t
                passing_options = [i]

        return passing_options
        '''
        unserved_players = list()
        players = [i for i in range(get_player_count())]
        for player in players:
            total_cells = np.sum(np.array([value for value in get_served()[player].values()]))
            if total_cells<=(self.turns*24):
                unserved_players.append(player)
        return unserved_players
        '''

    def cosine_similarity(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    def get_most_similar_player(self,get_player_count,get_served, player_idx, get_turns_recieved):
        players = [i for i in range(get_player_count())]
        players_feature_vector = list()
        for player in players:
            feature_vector = np.array([value + 0.001 for value in get_served()[player].values()]) # + 0.001 so we don't ever divide by 0
            feature_vector = feature_vector/(np.sum(feature_vector)+1)
            players_feature_vector.append(feature_vector)
        unserved_players = self.get_unserved_players(players,player_idx, get_turns_recieved)
        self_feature_vector = players_feature_vector[player_idx]

        most_similar_player = -1
        similarity = -1
        #print('unserved_players:',unserved_players)
        for unserved_player in unserved_players:
            if unserved_player==player_idx:
                continue
            feature_vector = players_feature_vector[unserved_player]
            cur_similarity = self.cosine_similarity(self_feature_vector,feature_vector)
            #print(cur_similarity)
            if cur_similarity>similarity:
                most_similar_player = unserved_player
                similarity = cur_similarity

        if most_similar_player == -1: #only ourself leave
            return player_idx
        else:
            return most_similar_player


        #def most_similar_top_preferences(self, get_player_count, get_served, player_idx, get_turns_recieved):




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
            self.state = 0 # reset for next turn

            values = self.get_most_similar_player(get_player_count,get_served, player_idx, get_turns_received)
            #print("Most similar player:",values)

            #self.turns+=1
            #values = player_idx
        else:
            action = "scoop"
            values = self.get_highest_score(top_layer,curr_level)
        return {"action": action,  "values": values}
