import os
import argparse
import numpy as np
import csv
import itertools
from tqdm import tqdm
from main import IceCreamGame, return_vals

ALL_PLAYERS_LIST = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
FLAVORS = [2, 3, 4, 6, 9, 12]
FAMILY_SIZES = [2, 3, 4, 6, 8, 9]
TRIALS = 10
RESULT_DIR = "results"
DEFAULT_SEED_ENTROPY = 233549467911963472626752780092567886323

seed_sequence = np.random.SeedSequence()

extra_df_cols = ["family_size", "flavors", "trial"]
all_df_cols = extra_df_cols+return_vals

def generate_args(flavors, log_path, seed):
    args = argparse.Namespace(address='127.0.0.1', automatic=False, disable_logging=True,disable_timeout=False, flavors=flavors, log_path=log_path, no_browser=False, no_gui=True, port=8080, seed=seed)
    return args

def worker(worker_input):
    family_size, player_list, flavors, trial = worker_input
    # print("Running with {} size family with members {} flavors {} trial {}".format(family_size, player_list, flavors, trial))
    seed = seed_sequence.generate_state(1, dtype=np.uint64)[0]
    log_path = "{}_size_family_{}_flavors_{}_trial_{}.log".format(family_size, "-".join(player_list), flavors, trial)
    log_path = os.path.join(RESULT_DIR, "logs", log_path)
    args = generate_args(flavors=flavors, log_path=log_path, seed=seed)
    ice_cream_game = IceCreamGame(player_list=player_list, args=args)
    ice_cream_game.play_all()
    result = ice_cream_game.get_state()
    for df_col in extra_df_cols:
        result[df_col] = eval(df_col)
    return result

if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)

    tournament_configs = []
    for family_size in FAMILY_SIZES:
        player_lists = itertools.combinations(ALL_PLAYERS_LIST,family_size)
        for player_list in player_lists:
            for flavors in FLAVORS:
                for trial in range(1, TRIALS+1):
                    config = (family_size, player_list, flavors, trial)
                    tournament_configs.append(config)

    out_fn = os.path.join(RESULT_DIR, "aggregate_results.csv")
    with open(out_fn, "w") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=all_df_cols)
        writer.writeheader()
        csvf.flush()
        for config in tqdm(tournament_configs):
            result = worker(config)
            writer.writerow(result)
            csvf.flush()