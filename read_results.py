import pandas as pd
import numpy as np

df = pd.read_csv("aggregate_results_family_sizes_2-3-4-6-8-9.csv", converters={"family_size": int, "flavors": int, "trial": int, "seed": int, "player_names": eval, "final_scores": eval, "winner_list": eval, "player_scores": eval, "player_preferences": eval, "served": eval, "total_time_sorted": eval, "turns_received": eval, "timeout_count": eval, "error_count": eval})

# print(df.values.tolist()[0])
# print(df.to_dict("records")[0])


# print(df.loc[0])
# for x in df.loc[0]:
#     print(x, type(x))
    # if isinstance(x, np.ndarray) and x.size:
    #     print(x.shape)
# sorted_times = df["total_time_sorted"]
# times = [x[1] for y in sorted_times for x in eval(y)]
