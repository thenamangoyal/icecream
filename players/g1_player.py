import logging
from typing import Callable, Dict, List, Tuple, Union, NamedTuple

import numpy as np


class Choice(NamedTuple):
    flavors: List[int]
    max_depth: int
    index: Tuple[int, int]


def generate_choices(top_layer: np.ndarray, curr_level: np.ndarray) -> List[Choice]:
    res = list()
    for i in range(top_layer.shape[0] - 1):
        for j in range(top_layer.shape[1] - 1):
            cur = list()
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
            res.append(Choice(cur, max_depth, (i, j)))
    return res


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
        self.logger.info(choices)
        if not choices:
            turns = get_turns_received()
            players = list()
            for idx, turn in enumerate(turns):
                if turn == min(turns):
                    players.append(idx)
            self.state.append(0)
            return dict(action="pass", values=self.rng.choice(players))

        def f(choice: Choice) -> float:
            res = 0
            for flavor in choice.flavors:
                res -= self.flavor_preference.index(flavor)
            res /= len(choice.flavors)
            res += choice.max_depth * 0.2
            return res
        choice = max(choices, key=f)
        self.state[-1] += len(choice.flavors)
        return dict(action='scoop', values=choice.index)
