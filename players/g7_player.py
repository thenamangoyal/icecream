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

        self.move_scores = defaultdict(list)
        self.round = 1
        self.priority_moves = [] # to store cumulative moves

        self.total_cells = 2880
        self.distribution = self.get_init_distribution()

        self.remaining_scoops = 24
        self.scoop_numbers = defaultdict(list)
        
        self.gaussian = { 12: [(18.93809223, 8.664641593), (16.77040656, 7.409357709), (14.33525924, 6.185959087), 
                              (11.52450328,	4.83889168), (9.123905629, 4.221469871), (7.079441935, 3.890010288),
                              (5.425096583, 3.578941249), (4.189042806, 3.24460244), (3.261668554, 2.972141223),
                              (2.555017409, 2.714515753), (2.015396434, 2.469860744), (1.662865898, 2.348139111)], 
                         9: [(41.02960009, 14.12029149), (30.42369231, 13.48909796), (16.19857437, 9.082679016), 
                             (7.480713647, 5.665023627), (3.048702418, 3.482439013), (1.1525718, 1.949435324), 
                             (0.428877479, 1.072511073), (0.165270926, 0.646891044), (0.071996962, 0.452972316)], 
                         6: [(54.72077126, 16.84742329), (28.96171214, 13.11718795), (10.64398781, 7.684032927), 
                             (3.484021653, 4.126773401), (1.33156718, 2.131334974), (0.857939953, 1.366120308)],
                         4: [(71.11783158, 16.44311349), (23.64762365, 13.53851539), (4.502956149, 6.220680334),
                             (0.73158863, 2.200507703)],
                         3: [(80.15432119, 15.14688763), (18.00458439, 13.4309277), (1.841094416, 4.224895053)], 
                         2: [(90.58839827, 12.80923962), (9.411601726, 12.80923962)] } 

        self.can_gaussian = True
        if len(flavor_preference) not in self.gaussian:
            self.can_gaussian = False

        self.player_prefs = {}
        
    def get_init_distribution(self):
        flavor_preference_len = len(self.flavor_preference)
        return [self.total_cells / flavor_preference_len] * flavor_preference_len

    def get_surface_flavors(self, top_layer):
        flavor_count = [0] * len(self.flavor_preference)
        for row in top_layer:
            for val in row:
                if val > 0:
                    flavor_count[val - 1] += 1
        return flavor_count

    def get_distribution(self, served, top_layer):
        # index of distribution corresopnds to flavor with value i + 1
        self.distribution = self.get_init_distribution()
        surface_counts = self.get_surface_flavors(top_layer)
        for index, value in enumerate(surface_counts):
            self.distribution[index] -= value
        for player in served:
            for flavor, consumed in player.items():
                self.distribution[flavor - 1] -= consumed

    def get_flavor_preferences(self, playerCount, numFlavors, served):
        gaussiansList = self.gaussian.get(numFlavors)
        estimatedPreferences = []
        for player in range(playerCount):
            playerPref = []
            bowl = served[player]
            totalCells = 0
            for count in bowl.values():
                totalCells += count
            for f in range(1, numFlavors + 1):
                if(totalCells == 0): #empty bowl
                    playerPref.append(0)
                    continue
                flavorCount = bowl.get(f)
                if flavorCount == 0: #no data on this flavor
                    playerPref.append(0)
                    continue
                percentage = (flavorCount*100)/totalCells
                vector = []
                for gaussian in gaussiansList:
                    mean = gaussian[0]
                    sigma = gaussian[1]
                    gOutput = (1/(sigma*math.sqrt(2*math.pi)))*math.exp((-1/2)*((percentage - mean)/sigma)**2)
                    vector.append(gOutput)

                #normalize vector
                norm = 0
                for gOutput in vector:
                    norm += gOutput
                #norm = math.sqrt(norm)
                vector = [gOutput/norm for gOutput in vector]

                #calculate expectation
                expectation = 0
                for i in range(len(vector)):
                    flavorValue = numFlavors - i
                    gOutput = vector[i]
                    expectation += flavorValue*gOutput

                #now expectation for this flavor is calculated, add to player preference list
                playerPref.append(expectation)
            #all flavors estimated for this player
            estimatedPreferences.append(playerPref)

        return estimatedPreferences

    def get_expected_user_score_of_surface_flavors(self, player_preferences, top_layer):
        # here we assume player_preferences returns [flavor_3, flavor_2, flavor_1], where the expected score
        # for each flavor is len(flavor_prefs) - index
        surface_flavor_count = self.get_surface_flavors(top_layer) # an array of flavors [flavor_1_ct, flavor_2_ct]
        n = len(player_preferences)
        expected_score = 0
        for i in range(n):
            flavorIdx = player_preferences[i]
            expected_score += surface_flavor_count[i] * flavorIdx
        return expected_score

    def get_best_pass(self, playerCount, top_layer, served, turns):
        bestPlayerToPass = None
        bestExpectedScore = -1
        max_turns = np.max(turns)
        min_turns = np.min(turns)
        for playerIdx in range(playerCount):
            if playerIdx == self.playerIdx or (turns[playerIdx] == max_turns and max_turns != min_turns):
                continue
            # print(self.player_count, len(self.flavor_preference), served)
            player_prefs = self.player_prefs[playerIdx]
            # print("hop", player_prefs)
            exp_score = self.get_expected_user_score_of_surface_flavors(player_prefs, top_layer)
            if exp_score > bestExpectedScore:
                bestExpectedScore = exp_score
                bestPlayerToPass = playerIdx
        return bestPlayerToPass

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

    def get_expected_user_score_of_scoop(self, scoop, player_prefs):
        total = 0
        for s in scoop:
            total += player_prefs[s-1]
        return total

    def updated_score(self, i, j, top_layer, curr_level, served):
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

        total_scoop = []

        for level in range(max_level, min_level - 1, -1):
            for x in range(2):
                for y in range(2):
                    if curr_level[i + x][j + y] == level:
                        cumulative_score += self.flavor_points[top_layer[i + x][j + y]]
                        total_scoop.append(top_layer[i + x][j + y])
                        num_scooped += 1
                        num_unseen_cells += 1

            num_moves += 1
            cumulative_score += prev_num_unseen_cells * self.flavor_points[-2]
            num_scooped += prev_num_unseen_cells
            score = (cumulative_score / num_scooped) if num_scooped > 0 else 0

            if score == 0:
                pass
            elif self.can_gaussian:
                best_for_others = -1
                for playerIdx in range(self.player_count):
                    if playerIdx == self.playerIdx:
                        continue
                    player_prefs = self.player_prefs[playerIdx]
                    exp_score = self.get_expected_user_score_of_scoop(total_scoop, player_prefs)

                    best_for_others = max(best_for_others, exp_score)
                
                score -= (best_for_others / num_scooped) / (self.player_count - 1) if num_scooped > 0 else 0
            
            self.move_scores[-score].append((num_moves, num_scooped, i, j))
            self.scoop_numbers[num_scooped].append((i,j))

            prev_num_unseen_cells = num_unseen_cells

    def should_not_take_scoop(self, scoop):
        # true means shouldn't take scoop, false means it can
        _, num_scooped, i, j = scoop
        if self.remaining_scoops == num_scooped or self.remaining_scoops > 8:
            return False
        elif (self.remaining_scoops - num_scooped) % 4 == 3:
            for s in self.scoop_numbers[3]:
                si, sj = s
                clear = False
                if i < si - 1 or i > si + 1:
                    clear = True
                if j < sj - 1 or j > sj + 1:
                    clear = True 
                if clear:
                    return False
        elif (self.remaining_scoops - num_scooped) % 4 == 2:
            for s in self.scoop_numbers[2]:
                si, sj = s
                clear = False
                if i < si - 1 or i > si + 1:
                    clear = True
                if j < sj - 1 or j > sj + 1:
                    clear = True 
                if clear:
                    return False
        elif (self.remaining_scoops - num_scooped) % 4 == 1:
            for s in self.scoop_numbers[1]:
                si, sj = s
                clear = False
                if i < si - 1 or i > si + 1:
                    clear = True
                if j < sj - 1 or j > sj + 1:
                    clear = True 
                if clear:
                    return False

        return True

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

        self.player_count = get_player_count()
        self.playerIdx = player_idx
        served = get_served()

        max_scoop_i, max_scoop_j = -1, -1
        max_scoop_point = -1

        self.update_hidden_cell_expectation(served, top_layer)

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
            if (self.can_gaussian and not self.player_prefs):
                for playerIdx in range(self.player_count):
                    if playerIdx == self.playerIdx:
                        continue
                    player_prefs = self.get_flavor_preferences(self.player_count, len(self.flavor_preference), served)[playerIdx]
                    self.player_prefs[playerIdx] = player_prefs
            
            for i in range(len(top_layer) - 1):
                for j in range(len(top_layer[i]) - 1):
                    self.updated_score(i, j, top_layer, curr_level, served)

            scores = list(self.move_scores.keys())

            if len(scores) == 0:
                time_to_pass = True
            else:
                heapq.heapify(scores)
                max_score = heapq.heappop(scores)
                self.move_scores[max_score].sort(key=lambda x: x[1], reverse=True)
                scoop = self.move_scores[max_score].pop(0)
                max_num_moves, max_scoop_cells, max_scoop_i, max_scoop_j = scoop

                while (max_scoop_cells > remaining_scoops or self.should_not_take_scoop(scoop)) and len(scores) > 0:
                    if len(self.move_scores[max_score]) == 0:
                        del self.move_scores[max_score]
                        max_score = heapq.heappop(scores)
                        self.move_scores[max_score].sort(key=lambda x: x[1], reverse=True)
                    scoop = self.move_scores[max_score].pop(0)
                    max_num_moves, max_scoop_cells, max_scoop_i, max_scoop_j = scoop

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
        self.move_scores = defaultdict(list)
        self.scoop_numbers = defaultdict(list)

        if time_to_pass or max_scoop_cells > remaining_scoops:
            if (self.can_gaussian):
                turns = get_turns_received()
                next_player = self.get_best_pass(get_player_count(), top_layer, served, turns)
                
                self.round += 1
                self.priority_moves = []
                self.remaining_scoops = 24
                self.player_prefs = {}
                return {"action": "pass", "values": next_player}
            else:
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
