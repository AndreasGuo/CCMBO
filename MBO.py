import random
from scipy.stats import levy
from numpy import array, empty, append, random, double, vectorize,Inf, zeros
from copy import copy
from nsga2 import dominates
from copy import deepcopy


class Individual:
    def __init__(self, mbo=None, positions=[], cost=0):
        if mbo is None:
            self.positions = array(copy(positions), dtype=double)
            self.cost = copy(cost)
        else:
            self.positions = random.rand(mbo.dimension) * (mbo.upper - mbo.lower) + mbo.lower
            self.cost = mbo.costFunction(self.positions)
        self.dominationSet = array([], dtype=int)
        self.dominatedCount = int(0)
        self.rank = Inf
        self.crowding_distance = 0

class MBO:
    def __init__(self, cost_function, sort_function, params):
        self.popSize = params['popSize']
        self.NP1 = params['NP1']
        self.NP2 = self.popSize - self.NP1
        self.maxGen = params['maxGen']
        self.S_max = params['S_max']
        # peri: migration period.
        self.peri = params['peri']
        # butterfly adjusting rate
        self.BAR = params['BAR']
        # migration ratio
        self.p = params['p']
        self.dimension = params['dimension']
        # defined as S_max/(t^2)
        self.alpha = params['alpha']
        self.lower = params['lower']
        self.upper = params['upper']
        self.costFunction = cost_function
        self.sortFunction = sort_function
        self.land1 = empty(self.NP1, dtype=Individual)
        self.new_land1 = copy(self.land1)
        self.land2 = empty(self.NP2, dtype=Individual)
        self.new_land2 = copy(self.land2)
        self.x_best = None
        self.t = 1
        self.pop = append(self.land1, self.land2)
        self.curve_fig = empty(self.popSize, dtype=double)

    def rectify_value(self, individuals): 
        if type(self.lower) in [double, float, int]:
            for i in individuals:
                rectify = lambda t: double(min(self.upper, max(self.lower, t)))
                vf = vectorize(rectify)
                i.positions = vf(i.positions)
        else:
            for i in individuals:
                for k in self.dimension:
                    i.positions[k] = min(self.upper[k], max(self.lower[k], i.positions[k]))

    def init_pop(self):
        for i in range(self.popSize):
            self.pop[i] = Individual(self)
            if self.x_best is None:
                self.x_best = deepcopy(self.pop[i]) #Individual(positions=self.pop[i].positions, cost=self.pop[i].cost)
            else:
                if dominates(self.pop[i].cost, self.x_best.cost):
                    self.x_best = Individual(positions=self.pop[i].positions, cost=self.pop[i].cost)

    def migrate(self):
        for i in range(self.NP1):
            for k in range(self.dimension):
                if random.randn() * self.peri <= self.p:
                    self.new_land1[i].positions[k] \
                        = self.land1[random.randint(self.NP1)].positions[k]
                else:
                    self.new_land1[i].positions[k] \
                        = self.land2[random.randint(self.NP2)].positions[k]

    def adjust(self):
        for i in range(self.NP2):
            for k in range(self.dimension):
                rand = random.randn()
                if rand <= self.p:
                    self.new_land2[i].positions[k] \
                        = self.x_best.positions[k]
                else:
                    self.new_land2[i].positions[k] \
                        = self.land2[random.randint(self.NP2)].positions[k]
                    if rand > self.BAR:
                        dx = levy.rvs(self.land2[i].positions[k])
                        self.new_land2[i].positions[k] \
                            = self.new_land2[i].positions[k] \
                            + self.S_max / (self.t ** 2) * (dx - 0.5)

    def calculate_cost(self):
        self.rectify_value(self.pop)
        for i in range(self.popSize):
            self.pop[i].cost = self.costFunction(self.pop[i].positions)

    def iterate(self):
        while self.t <= self.maxGen:
            self.pop = self.sortFunction(self.pop)
            self.land1, self.land2 = self.pop[0:self.NP1], self.pop[self.NP1:]
            self.new_land1 = copy(self.land1)
            self.new_land2 = copy(self.land2)
            self.migrate()
            self.adjust()
            self.pop = append(self.new_land1, self.new_land2)
            self.calculate_cost()
            self.pop = self.sortFunction(self.pop)
            if self.x_best.cost > self.pop[0].cost:
                self.x_best = Individual(positions=self.pop[0].positions, cost = self.pop[0].cost)
            self.curve_fig[self.t-1] = self.x_best.cost
            print(f'x_best.positions: {self.x_best.positions}, cost: {self.x_best.cost}')
            self.t = self.t+1

    def boot(self):
        self.init_pop()
        self.iterate()
        return self.x_best, self.curve_fig

