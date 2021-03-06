import os
import argparse
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from main import IceCreamGame, return_vals
from multiprocessing import Pool


def generate_args(flavors, log_path, seed):
    args = argparse.Namespace(address='127.0.0.1', automatic=False, disable_logging=True, disable_timeout=False, flavors=flavors, log_path=log_path, no_browser=False, no_gui=True, port=8080, seed=seed)
    return args


def worker(worker_input):
    global RESULT_DIR, extra_df_cols
    family_size, player_list, flavors, trial, seed = worker_input
    # print("Running with {} size family with members {} flavors {} trial {} seed {}".format(family_size, player_list, flavors, trial, seed))
    log_path = None
    args = generate_args(flavors=flavors, log_path=log_path, seed=seed)
    ice_cream_game = IceCreamGame(player_list=player_list, args=args)
    ice_cream_game.play_all()
    result = ice_cream_game.get_state()
    for df_col in extra_df_cols:
        result[df_col] = eval(df_col)
    return result


def get_player_lists(family_size, repeat_count):
    player_lists = []
    if family_size % repeat_count == 0:
        m = family_size//repeat_count
        base_player_lists = list(itertools.combinations(ALL_PLAYERS_LIST, m))
        repeat_player_lists = [base_player_list*repeat_count for base_player_list in base_player_lists]
        if len(repeat_player_lists) > 0:
            player_lists += repeat_player_lists
    return player_lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="results", help="Directory path to dump results")
    parser.add_argument("--seed_entropy", "-s", type=int, help="Seed used to generate seed for each game")
    parser.add_argument("--family_size", "-f", default=[2, 3, 4, 6, 8, 9, 12], nargs="+", type=int, help="List of size of family")
    args = parser.parse_args()
    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)
    FAMILY_SIZES = args.family_size

    ALL_PLAYERS_LIST = ["1", "2", "3", "4", "5", "7", "8", "9", "10"]
    FLAVORS = [2, 3, 4, 6, 9, 12]
    REPEAT_COUNTS = [2, 3, 4, 6, 12]
    TRIALS = 10

    extra_df_cols = ["family_size", "flavors", "trial", "seed"]
    all_df_cols = extra_df_cols+return_vals

    print("Using family sizes {}".format(FAMILY_SIZES))
    seed_sequence = np.random.SeedSequence(args.seed_entropy)
    print("Using seed sequence with entropy {}".format(seed_sequence.entropy))
    with open(os.path.join(RESULT_DIR, "config.txt"), "w") as f:
        f.write("Family sizes {}\n".format(FAMILY_SIZES))
        f.write("Seed entropy {}\n".format(seed_sequence.entropy))

    base_tournament_configs = dict()
    for family_size in FAMILY_SIZES:
        base_tournament_configs[family_size] = []
        if family_size <= len(ALL_PLAYERS_LIST):
            times_to_repeat = [1, family_size]
        else:
            times_to_repeat = REPEAT_COUNTS
        player_lists = []
        for repeat_count in times_to_repeat:
            player_lists = get_player_lists(family_size, repeat_count)
            for player_list in player_lists:
                for flavors in FLAVORS:
                    for trial in range(1, TRIALS+1):
                        base_config = (family_size, player_list, flavors, trial)
                        base_tournament_configs[family_size].append(base_config)

    config_lens = {family_size:len(base_tournament_configs[family_size]) for family_size in FAMILY_SIZES}
    print("No of configs to run by family size\n{}".format(config_lens))
    print("Total configs {}".format(sum(config_lens.values())))
    seeds = seed_sequence.generate_state(sum(config_lens.values()), dtype=np.uint64)

    tournament_configs = []
    for family_size in FAMILY_SIZES:
        for i, base_config in enumerate(base_tournament_configs[family_size]):
            config = tuple(list(base_config) + [seeds[i]])
            tournament_configs.append(config)

    out_fn = os.path.join(RESULT_DIR, "aggregate_results_family_sizes_{}.csv".format("-".join(list(map(str, FAMILY_SIZES)))))
    with open(out_fn, "w") as csvf:
        header_df = pd.DataFrame([], columns=all_df_cols)
        header_df.to_csv(csvf, index=False, header=True)
        csvf.flush()
        with Pool() as p:
            for result in tqdm(p.imap(worker, tournament_configs), total=len(tournament_configs)):
                df = pd.DataFrame([result], columns=all_df_cols)
                df.to_csv(csvf, index=False, header=False)
                csvf.flush()
