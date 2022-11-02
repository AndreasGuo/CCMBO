import numpy as np
from concurrent.futures import ThreadPoolExecutor

pool = ThreadPoolExecutor(max_workers=10)

def dominates(x, y):
    return np.sum(x<=y) == len(x)

# calculate rank only and no order changed.
def paretoSorting(individuals):
    rankSet = list()
    # rank 1 which is not dominated by other individuals.
    for i in range(len(individuals)):
        j=i 
        while j<len(individuals)-1:
            j+=1
            if dominates(individuals[i].cost, individuals[j].cost):
                individuals[i].dominationSet = np.append(individuals[i].dominationSet, j)
                individuals[j].dominatedCount += 1
            elif dominates(individuals[j].cost, individuals[i].cost):
                individuals[j].dominationSet = np.append(individuals[j].dominationSet, i)
                individuals[i].dominatedCount += 1

        if individuals[i].dominatedCount == 0:
            individuals[i].rank = 1
            if len(rankSet)==0:
                rankSet.append([])
            rankSet[0].append(i)
    
    # rank 2...n
    k = 1
    while True: 
        Q = list()
        for i in rankSet[k-1]:
            p = individuals[i]
            for j in p.dominationSet:
                q = individuals[j]
                q.dominatedCount -= 1
                if q.dominatedCount == 0:
                    Q.append(j)
                    q.rank = k+1
                individuals[j] = q
        if len(Q) == 0:
            break 
        k += 1
        rankSet.append(Q)
    return rankSet
            
# Copyright (c) 2015, Mostapha Kalami Heris & Yarpiz (www.yarpiz.com)
# copy from matlab, but I think he never dive into the Costs = [pop(F{k}).Cost]
# it just return 1*n dim, the size(Costs, 1) is dispensible.

def crowding_distance(individuals, rankSet):
    dimension = len(individuals[0].cost)
    def cd_one_set(ri):
        rset = rankSet[ri]
        r_set_len = len(rset)
        distances = np.zeros([r_set_len, dimension])
        for d in range(dimension):
            costs = np.array([])
            # the pop index in the set which share one rank.
            for rset_i in range(len(rset)):
                pi = individuals[rset[rset_i]]
                costs = np.append(costs, pi.cost[d])
            arg_order = np.argsort(costs)
            distances[arg_order[0], d] = np.Inf 
            for rset_i in range(1,len(rset)-1):
                width = abs(costs[arg_order[0]] - costs[arg_order[-1]])
                if width == 0:
                    distances[arg_order[rset_i], d] = np.Inf
                else:
                    distances[arg_order[rset_i], d] = \
                        abs(costs[arg_order[rset_i-1]] - costs[arg_order[rset_i+1]]) / width
                        
            distances[arg_order[-1], d] = np.Inf
        
        for i in range(r_set_len):
            individuals[rset[i]].crowding_distance = sum(distances[i])
    
    pool.map(cd_one_set, np.arange(len(rankSet)))
    #for ri in range(len(rankSet)):
        

def clear_nsga(individuals):
    for i in individuals:
        i.dominationSet = np.array([], dtype=int)
        i.dominatedCount = int(0)
        i.rank = np.Inf
        i.crowding_distance = 0

def nsga2(individuals):
    clear_nsga(individuals)
    rankSet = paretoSorting(individuals)
    crowding_distance(individuals, rankSet)

