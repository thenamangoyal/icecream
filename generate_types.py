import os
import shutil
import numpy as np
import json

rng = np.random.default_rng(2021)

flavors = list(range(1,13))
l = 24 # cols
w = 15 # rows
h = 8 # height
container = np.empty((l, w, h), dtype=np.int)

assert container.size % len(flavors) == 0, "Number of flavors can't divide container size"
flavor_assigned = np.repeat(np.array(flavors, dtype=np.int), container.size/ len(flavors))

rng.shuffle(flavor_assigned)

dirname = "types"

if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.makedirs(dirname)

for ice_cream_type in [2,3,4,5,6,8,9,10,12]:
    with open(os.path.join(dirname, "{}.json".format(ice_cream_type)),"w") as jf:
        rng.shuffle(flavor_assigned)
        json.dump(np.reshape(flavor_assigned, (l, w, h)).tolist(), jf)