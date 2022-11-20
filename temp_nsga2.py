import numpy as np 
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

class NSGA2():
    def __init__(self, individuals):
        self.individuals = individuals
        pass 

    def dominates(x, y):
        return np.sum(x<=y) == len(x)

    def paretoSorting(self):
        rankSet = list()
        # rank 1 which is not dominated by other individuals.
        for i in range(len(self.individuals)):
            j=i 
            while j<len(self.individuals)-1:
                j+=1
                if self.dominates(self.individuals[i].cost, self.individuals[j].cost):
                    self.individuals[i].dominationSet = np.append(self.individuals[i].dominationSet, j)
                    self.individuals[j].dominatedCount += 1
                elif self.dominates(self.individuals[j].cost, self.individuals[i].cost):
                    self.individuals[j].dominationSet = np.append(self.individuals[j].dominationSet, i)
                    self.individuals[i].dominatedCount += 1

            if self.individuals[i].dominatedCount == 0:
                self.individuals[i].rank = 1
                if len(rankSet)==0:
                    rankSet.append([])
                rankSet[0].append(i)
        
        # rank 2...n
        k = 1
        while True: 
            Q = list()
            for i in rankSet[k-1]:
                p = self.individuals[i]
                for j in p.dominationSet:
                    q = self.individuals[j]
                    q.dominatedCount -= 1
                    if q.dominatedCount == 0:
                        Q.append(j)
                        q.rank = k+1
                    self.individuals[j] = q
            if len(Q) == 0:
                break 
            k += 1
            rankSet.append(Q)
        return rankSet
