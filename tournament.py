import os
import argparse
import numpy as np
import pandas as pd
import itertools
from main import IceCreamGame, return_vals

def generate_args(flavors, log_path, seed):
    args = argparse.Namespace(address='127.0.0.1', automatic=False, disable_logging=True,disable_timeout=False, flavors=flavors, log_path=log_path, no_browser=False, no_gui=True, port=8080, seed=seed)
    return args

ALL_PLAYERS_LIST = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
FLAVORS = [2, 3, 4, 6, 9, 12]
FAMILY_SIZES = [2, 3, 4, 6, 8, 9]
TRIALS = 10
RESULT_DIR = "results"
DEFAULT_SEED_ENTROPY = 233549467911963472626752780092567886323

seed_sequence = np.random.SeedSequence()

os.makedirs(RESULT_DIR, exist_ok=True)

df_cols = ["family_size", "flavors", "trial"]

results = []
for family_size in FAMILY_SIZES:
    player_lists = itertools.combinations(ALL_PLAYERS_LIST,family_size)
    for player_list in player_lists:
        for flavors in FLAVORS:
            for trial in range(1, TRIALS+1):
                print("Running with {} size family with members {} flavors {} trial {}".format(family_size, player_list, flavors, trial))
                seed = seed_sequence.generate_state(1, dtype=np.uint64)[0]
                log_path = "{}_size_family_{}_flavors_{}_trial_{}.log".format(family_size, "-".join(player_list), flavors, trial)
                log_path = os.path.join(RESULT_DIR, "logs", log_path)
                args = generate_args(flavors=flavors, log_path=log_path, seed=seed)
                ice_cream_game = IceCreamGame(player_list=player_list, args=args)
                ice_cream_game.play_all()
                result = ice_cream_game.get_state()
                for df_col in df_cols:
                    result[df_col] = eval(df_col)
                results.append(result)

if len(results) > 0:
    results_df = pd.DataFrame(results, columns=df_cols+return_vals)
    out_fn = os.path.join(RESULT_DIR, "aggregate_results.csv")
    print("Saving results from {} runs to {}".format(len(results_df), out_fn))
    results_df.to_csv(out_fn, index=False)