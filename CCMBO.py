from MBO import MBO
from numpy import empty, append, random, double, cov, zeros, arange, vstack, std, ceil, sqrt, vectorize, average
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from scipy.stats import levy
from sklearn_som.som import SOM
import time
from nsga2 import dominates

# 因为有O的原故，所以upper与lower必为数值类型 而不能是数组
# cause O is one dimension array, upper and lower must be number rather than array.

class CCMBO(MBO):
    def __init__(self, cost_function, sort_function, params):
        MBO.__init__(self, cost_function, sort_function, params)
        self.epsilon = params['epsilon']  # threshold of re-initialization, compare to standard deviation of the best fitness.
        self.C_r = params['C_r']

    def cov(self, j_array, k_array):
        j_len = len(j_array)
        k_len = len(k_array)
        j_avg = average(j_array)
        k_avg = average(k_array)

    def cov_jk(self, land, j, k):
        if land == 1:
            nland = self.land1 
        else:
            nland = self.land2
        j_array = empty(len(nland), dtype=double)
        k_array = empty(len(nland), dtype=double)
        for i in nland:
            j_array[j] = i.positions[j]
            k_array[k] = i.positions[k]
        return cov(j_array, k_array)[0,1]

    def cov_land_matrix(self, land):
        pool = ThreadPoolExecutor(max_workers=10)
        size = self.dimension
        cov_land = zeros([size, size])
        j_array = arange(0, self.dimension)
        k_array = arange(0, self.dimension)
        iterable_land = zeros(size, dtype=int) + land
        result_iterators = pool.map(self.cov_jk, iterable_land, j_array, k_array)
        i = 0
        for result in result_iterators:
            cov_land[int(i / size), i % size] = result
            i += 1
        return cov_land

    def positions(self):
        positions1 = self.land1[0].positions
        i = 1
        while i < self.NP1:
            positions1 = vstack((positions1, self.land1[i].positions)) 
            i += 1

        positions2 = self.land2[0].positions
        i = 1
        while i < self.NP2:
            positions2 = vstack((positions2, self.land2[i].positions)) 
            i += 1
        return positions1, positions2

    def cov_matrix(self):
        # calculate land1's Cov Matrix
        cov_land1 = self.cov_land_matrix(1)

        # calculate land2's Cov Matrix
        cov_land2 = self.cov_land_matrix(2)

        return cov_land1, cov_land2

    def migrate(self, eig_land1, eig_land2, q_land1, q_land2):
        for i in range(self.NP1):
            eig_temp1 = deepcopy(eig_land1[i, :])
            eig_temp2 = deepcopy(eig_land1[i, :])
            if random.rand() <= 0.5:
                for j in range(self.dimension):
                    if random.rand() < self.p:
                        rand = random.randint(self.NP1)
                        eig_temp1[j] = eig_land1[rand, j]
                    else:
                        rand = random.randint(self.NP2)
                        eig_temp1[j] = eig_land2[rand, j]

                temp1 = eig_temp1.dot(q_land1.T)
                temp2 = eig_temp2.dot(q_land2.T)
                cost1 = self.costFunction(temp1)
                cost2 = self.costFunction(temp2)
                if dominates(cost1, cost2):
                    self.new_land1[i].positions = temp1
                    self.new_land1[i].cost = cost1
                else:
                    self.new_land1[i].positions = temp2
                    self.new_land1[i].cost = cost2
            else:
                MBO.migrate(self)

    def adjust(self, alpha):
        for j in range(self.NP2):
            beta = 0.5
            rand = random.randn()
            if rand <= self.p:
                self.new_land2[j].positions \
                    = self.land2[j].positions \
                      + beta * (self.x_best.positions - self.land2[j].positions)
            else:
                self.new_land2[j].positions = self.land2[random.randint(self.NP2)].positions
                if rand > self.BAR:
                    dx = levy.rvs(self.land2[j].positions)
                    self.new_land2[j].positions += alpha * (dx - 0.5)

    def reinitializing_land2(self, delta):
        # delta is the standard deviation of the best fitness.
        if delta < self.epsilon:
            for j in range(self.NP2):
                rand = random.rand()
                if rand < 0.5:
                    for k in range(self.dimension):
                        self.new_land2[j].positions[k] = self.lower + (self.upper - self.lower) * rand
                else:
                    for k in range(self.dimension):
                        self.new_land2[j].positions[k] = self.lower + self.upper - self.new_land2[j].positions[k]

    def calculate_new_pop_cost(self, new_pop):
        self.rectify_value(new_pop)
        for i in range(self.popSize):
            new_pop[i].cost = self.costFunction(self.pop[i].positions)

    def iterate(self):
        while self.t <= self.maxGen:
            print(f"iteration {self.t} of {self.maxGen}")
            start_time = time.time()
            self.pop == self.sortFunction(self.pop)
            self.land1, self.land2 = deepcopy(self.pop[0:self.NP1]), deepcopy(self.pop[self.NP1:])
            self.new_land1 = deepcopy(self.land1)
            self.new_land2 = deepcopy(self.land2)
            # eig_land1, eig_land2, q_land1, q_land2
            q_land1, q_land2 = self.cov_matrix()
            positions1, positions2 = self.positions()
            eig_land1 = positions1.dot(q_land1)
            eig_land2 = positions2.dot(q_land2)
            self.migrate(eig_land1, eig_land2, q_land1, q_land2)
            self.adjust(self.S_max / (self.t ** 2))
            new_pop = append(self.new_land1, self.new_land2)
            self.calculate_new_pop_cost(new_pop)
            new_pop = self.sortFunction(new_pop)
            if random.rand() < self.C_r:
                M = random.randint(2, ceil(sqrt(self.popSize)))
                cluster = deepcopy(new_pop[0].positions)
                for j in random.choice(self.popSize,M,replace=False):
                    cluster = vstack((cluster,deepcopy(new_pop[j].positions)))
                iris_som = SOM(m=self.popSize, n=1, dim=self.dimension)
                iris_som.fit(cluster)
                O = iris_som.predict(cluster)

                # rectify O
                rectify = lambda t: min(self.upper, max(self.lower, t))
                vf = vectorize(rectify)
                O = vf(O)
                chosen = random.default_rng().choice(self.popSize, size=M, replace=False)
                chosen2 = random.default_rng().choice(M, size=M, replace=False)
                # 从保护种群数量的角度说，应是从O中选出M个，换掉pop中的M个
                # 不是上面那样, SOM出来的不是m*n的，而是m*1的
                for i in range(len(chosen)):
                    new_pop[chosen[i]].position = deepcopy(O[chosen2[i]])
                self.calculate_new_pop_cost(new_pop)
                new_pop = self.sortFunction(new_pop)
            # if re-initialization condition is true
            # TODO: 这里在多目标函数下可能会有问题
            delta = std(self.curve_fig[0:self.t-1])
            self.reinitializing_land2(delta)
            # combine
            self.land1 = deepcopy(self.new_land1)
            self.land2 = deepcopy(self.new_land2)
            self.pop = append(self.land1, self.land2)
            self.calculate_cost()
            self.pop = self.sortFunction(self.pop)
            if dominates(self.pop[0].cost, self.x_best.cost):
                self.x_best = deepcopy(self.pop[0])
            self.curve_fig[self.t-1] = sum(self.x_best.cost)
            print("best cost is {0}, time={1:.2f}".format(self.x_best.cost, (time.time() - start_time)))
            self.t += 1
