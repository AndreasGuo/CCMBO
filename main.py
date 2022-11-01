from numpy import log
from numpy import arange
from functools import cmp_to_key
from MBO import MBO
from CCMBO import CCMBO
import matplotlib.pyplot as pl

params={
    'popSize': 300,
    'NP1': 180,
    'maxGen': 300,
    'S_max': 1,
    'peri' : 1.2,
    'BAR': 0.3,
    'p': 5/12,
    'dimension': 2,
    'alpha': 0.0,
    'lower': -2,
    'upper': 2,
    'epsilon': 0.2,
    'C_r': 0.1
}
def cost_function(a):
    return a[0]*a[0] + a[1]*a[1]


def cmp_by_cost(x,y):
    if x.cost > y.cost:
        return 1
    else:
        return -1
    return 0


def sort_function(a):
    return sorted(a, key=cmp_to_key(cmp_by_cost))

mbo = CCMBO(cost_function, sort_function, params)
best, curve = mbo.boot()
print(best.positions)
pl.plot(arange(mbo.maxGen), curve)
pl.title('cost')
pl.show()