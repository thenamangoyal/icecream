import math
import numpy as np
from scipy import spatial
import copy
import logging

def NormalizeData(data):
    if np.max(data) == 0 : return list(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def passToMatch(self, top_layer, served, turns_received):
    unique, counts = np.unique(top_layer, return_counts=True)
    if -1 in top_layer:
        unique = np.delete(unique, 0)
        counts = np.delete(counts, 0)
    normalized_top_falvors = NormalizeData(counts)

    normalized_served = []
    distances = []
    for dict in served:
        norm_values = NormalizeData(np.fromiter(dict.values(), dtype=int))
        normalized_served.append(norm_values)
        distances.append(np.linalg.norm(norm_values-normalized_top_falvors))

    min_dist = min(distances)
    min_idx = distances.index(min_dist)
    idx = min_idx
    cut = distances[:]
    max_turn =  max(turns_received)
    while turns_received[min_idx] == max_turn and max_turn!=min(turns_received):
        cut = cut[:idx]+cut[idx+1:]
        if len(cut) == 0: break
        min_dist = min(cut)
        min_idx = distances.index(min_dist)
        idx = cut.index(min_dist)
    
    if len(cut) > 1 and min(cut) == max(cut):
        indices = [i for i, x in enumerate(distances) if x == min(cut)]
        min_idx = indices[self.rng.integers(0, len(cut))]

    # print(normalized_served)
    # tree = spatial.KDTree(normalized_served)
    # _, idx = tree.query(normalized_top_falvors)

    return min_idx

