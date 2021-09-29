import os
import shutil
import numpy as np
import json

rng = np.random.default_rng(2021)

flavors = list(range(1,13))
l = 24 # cols
w = 15 # rows
h = 8 # height
empty_container = np.empty((l, w, h), dtype=np.int)

assert empty_container.size % len(flavors) == 0, "Number of flavors can't divide container size"
flavor_repeated = np.repeat(np.array(flavors, dtype=np.int), empty_container.size/ len(flavors))

dirname = "types"

if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.makedirs(dirname)

for ice_cream_type in [2,3,4,5,6,8,9,10,12]:
    with open(os.path.join(dirname, "{}.json".format(ice_cream_type)),"w") as jf:
        flavor_assigned = rng.permutation(flavor_repeated)
        json.dump(np.reshape(flavor_assigned, empty_container.shape).tolist(), jf)