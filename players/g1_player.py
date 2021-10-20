import logging
import math
from typing import Callable, Dict, List, Tuple, Union, NamedTuple

import numpy as np
from collections import defaultdict


class Choice(NamedTuple):
    flavors: List[int]
    max_depth: int
    index: Tuple[int, int]
    all_cells: List[Tuple[int, int]]


def generate_choices(top_layer: np.ndarray, curr_level: np.ndarray) -> List[Choice]:
    res = list()
    for i in range(top_layer.shape[0] - 1):
        for j in range(top_layer.shape[1] - 1):
            cur = list()
            all_cells = list()
            max_depth = -1
            for x in range(2):
                for y in range(2):
                    max_depth = max(max_depth, curr_level[i + x][j + y])
            if max_depth == -1:
                continue
            for x in range(2):
                for y in range(2):
                    if max_depth == curr_level[i + x][j + y]:
                        cur.append(top_layer[i + x][j + y])
                        all_cells.append((i + x, j + y))
            res.append(Choice(cur, max_depth, (i, j), all_cells))
    return res


def choose_next_player(now_turn: int, possible_players: List[int], served_situation: List[Dict[int, int]],
                       top_layer: np.ndarray, curr_level: np.ndarray, rng: np.random.Generator,
                       my_player_idx: int, my_flavor_preference: List[int], total_players: int) -> int:
    if len(possible_players) == 1:
        return possible_players[0]

    # if now_turn = 0, it means for the first turn, we just randomly choose a person to pass to
    if now_turn == 0 and len(possible_players) != 0:
        return rng.choice(possible_players)

    # if there are no candidates, which means our team is the last one to serve
    if len(possible_players) == 0:
        possible_players = [i for i in range(total_players)]

    max_turns = math.floor(120 / total_players)

    choices = generate_choices(top_layer, curr_level)
    # if all ice cream has been taken
    if len(choices) == 0:
        return rng.choice(list(possible_players))

    greedy_flavor_preference = defaultdict(list)
    for player_index in possible_players:
        player_served_situation = sorted(served_situation[player_index].items(), key=lambda item: -item[1])
        for flavor, _ in player_served_situation:
            greedy_flavor_preference[player_index].append(flavor)

    max_score = -math.inf
    next_player_list = []
    # the more groups, the less choices
    total_units = 24
    if total_players > 4:
        total_units = 24 - int((0.3 * total_players) * 4)

    weighted = weightedFunction(now_turn, max_turns)

    for player_index, flavor_preference in greedy_flavor_preference.items():
        score_list = []
        for choice in choices:
            score_list.append((score(choice, flavor_preference), choice.all_cells, len(choice.flavors)))

        score_list.sort(key=lambda x: -x[0])

        remain = total_units - score_list[0][2]
        player_max_score = score_list[0][0]

        before_choice = score_list[0][1]

        for score_num, current_choice, count in score_list[1:]:
            overlap = findOverlapScoop(before_choice, current_choice)
            if remain - (count - overlap) < 0:
                break
            remain -= (count - overlap)
            player_max_score += score_num

        player_max_score = player_max_score / (total_units - remain) \
                           + weighted * difference(served_situation[my_player_idx], served_situation[player_index])

        if max_score < player_max_score:
            max_score = player_max_score
            next_player_list = [player_index]
        elif max_score == player_max_score:
            next_player_list.append(player_index)

    # if len(next_player_list) == 1, just return,
    # if the length is larger than 1
    if len(next_player_list) == 1:
        return list(next_player_list)[0]

    flavor_differences = []
    for player_index in next_player_list:
        flavor_differences.append(
            (difference(served_situation[my_player_idx], served_situation[player_index]), player_index))
    flavor_differences.sort(key=lambda x: x[0])
    return flavor_differences[0][1]


def weightedFunction(turn, max_turns):
    # we trust the later information more than before,
    # but the last information may be trusted less because groups have fewer choices
    exp_fun = 1 - math.exp(- turn / max_turns * 5)
    if turn / max_turns > 0.8:
        exp_fun -= 0.1 * turn / max_turns
    return exp_fun


def findOverlapScoop(choice1: List[Tuple[int, int]], choice2: List[Tuple[int, int]]) -> int:
    overlap = 0
    for scoopi, scoopj in choice2:
        if (scoopi, scoopj) in choice1:
            overlap += 1
    return overlap


# copy from the below f function
def score(choice: Choice, flavor_preference) -> float:
    res = 0
    for flavor in choice.flavors:
        res -= flavor_preference.index(flavor)
    res /= len(choice.flavors)
    res += choice.max_depth * 0.2
    res += 0.01 * len(choice.flavors)
    return res


def difference(my_flavor_preference: Dict[int, int], other_flavor_preference: Dict[int, int]) -> float:
    sum_difference = 0.0
    my_flavor_preference_count = sum(my_flavor_preference.values())
    other_flavor_preference_count = sum(other_flavor_preference.values())
    for flavor, flavor_count in my_flavor_preference.items():
        sum_difference += abs(flavor_count / my_flavor_preference_count -
                              other_flavor_preference[flavor] / other_flavor_preference_count)
    return sum_difference / len(my_flavor_preference)


class Player:
    def __init__(self, flavor_preference: List[int], rng: np.random.Generator, logger: logging.Logger) -> None:
        self.flavor_preference = flavor_preference
        self.rng = rng
        self.logger = logger
        self.state = [0]

    def serve(self, top_layer: np.ndarray, curr_level: np.ndarray, player_idx: int,
              get_flavors: Callable[[], List[int]], get_player_count: Callable[[], int],
              get_served: Callable[[], List[Dict[int, int]]], get_turns_received: Callable[[], List[int]]
              ) -> Dict[str, Union[Tuple[int, int], int]]:
        remain = 24 - self.state[-1]
        choices = generate_choices(top_layer, curr_level)
        choices = list(filter(lambda x: len(x.flavors) <= remain, choices))
        # self.logger.info(choices)
        if not choices:
            turns = get_turns_received()
            total_players = get_player_count()
            players = list()
            for idx, turn in enumerate(turns):
                if turn == min(turns) and idx != player_idx:
                    players.append(idx)
            self.state.append(0)
            # if we choose the person with the highest score
            next_player = choose_next_player(min(turns), players, get_served(), top_layer, curr_level, self.rng,
                                             player_idx, self.flavor_preference, total_players)
            return dict(action="pass", values=next_player)

            # if we just randomly choose one person
            # if len(players) == 0:
            #     players = [i for i in range(total_players)]
            # return dict(action="pass", values=self.rng.choice(players))

        def f(choice: Choice) -> float:
            res = 0
            for flavor in choice.flavors:
                res -= self.flavor_preference.index(flavor)
            res /= len(choice.flavors)
            # it seems that max_depth does have a positive impact on scoring after testing,
            # but we should still test weight = 0.2
            res += choice.max_depth * 0.2
            res += 0.01 * len(choice.flavors)
            return res

        choice = max(choices, key=f)
        self.state[-1] += len(choice.flavors)
        return dict(action='scoop', values=choice.index)
