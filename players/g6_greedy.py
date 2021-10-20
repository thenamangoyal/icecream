import math
import numpy as np
import copy
import logging
from typing import Callable, Dict, List, Tuple, Union
def get_level(self, curr_level,i,j):
    highest_level = 0
    for x in range(i,i+2):
        for y in range(j,j+2):
            if curr_level[x][y]>highest_level:
                highest_level = curr_level[x][y]
    return highest_level

def get_icecream(self,top_layer,curr_level,level,i,j):
    value = 0
    amount = 0
    if level<=0:
        return 0,0
    for x in range(i,i+2):
        for y in range(j,j+2):
            if curr_level[x][y]==level and top_layer[i][j] in self.flavor_preference:
                amount+=1
                value+=(len(self.flavor_preference)+1-self.flavor_preference.index(top_layer[i][j]))
    return value, amount
def greedy(self, top_layer, curr_level,maxAmount):
    bestValue = 0.0 # best mean value of ice-cream
    bestPos=(-1,-1) # position of bestValue
    bestAmount=0 # amount of ice-cream of bestValue
    for i in range(23):
        for j in range(14):
            level = get_level(self,curr_level,i,j)
            value,amount = get_icecream(self,top_layer,curr_level,level,i,j)
            if (amount>0 and amount<=maxAmount):
                if (float(value/amount)>bestValue):
                    bestAmount = amount
                    bestValue = float(value/amount)
                    bestPos = (i,j)
    print(bestValue*bestAmount)
    return bestPos, bestAmount