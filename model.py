"""Module works with the model according to it's initial parameters"""
from agent import Agent
import random as r
from numpy.random import normal

r.seed(1000)


class Model:
    """
    Model manipulations

    T - number of current iteration
    a* - confidence parameter of each agent
    x_hat - goal opinion for each agent
    opinions_dict - dictionary of model evolution like {T : [Agent0.opinion, Agent1.opinion, ...]}
    agents_list - list of Agents as elements of class with current opinions
    control - list of control value, index = T
    stopped - indicates is model stopped evolve
    a - list of confidence level of each agent
    exitcode -
    1 - reaching the goal
    2 - too many iterations
    3 - too big opinions
    4 - control value = 0
    5 - no sufficient opinions changes
    """
    T = 0
    stopped = False
    err = 0.03
    max_delta = 1

    # Initializes model with agents
    def __init__(self, input_data, adj_matrix, init_ops):
        """
        Init the object of class Model

        Fill all class fields with needed data
        """
        self.opinions_dict = dict()
        self.agents_list = [Agent(init_ops[0], 0, False),
                            Agent(init_ops[1], 1, True),
                            Agent(init_ops[2], 2, False)]
        self.opinions_dict[self.T] = [i.opinion for i in self.agents_list]
        self.a1 = input_data['a1']
        self.a2 = input_data['a2']
        self.a3 = input_data['a3']
        self.a = [self.a1, self.a2, self.a3]
        self.conf_matrix = [[1 - 2 * self.a1 / 3, self.a1 / 3, self.a1 / 3],
                            [self.a2 / 3, 1 - 2 * self.a2 / 3, self.a2 / 3],
                            [self.a3 / 3, self.a3 / 3, 1 - 2 * self.a3 / 3]]
        self.adj_matrix = adj_matrix
        self.x_hat = input_data['x_hat']
        self.control = []

    def pareto(self):
        # r.randrange(-1, 2, step = 2) *
        return r.randrange(-1, 2, step = 2) * (0.1 * pow(0.001, 0.1) * pow(self.T + 1, -0.1 - 1))\
               / (1 - pow(0.01/1, 0.1))

    def __adj_agents(self, n):
        """Return all indexes of adjacent agents for current agent"""
        adj = []
        for i in range(0, len(self.adj_matrix)):
            if self.adj_matrix[n][i] != 0:
                adj.append(i)
        return adj

    def mean_for_adj(self, n):
        """Return mean among adjacent agents for agent with index = n"""
        adj = self.__adj_agents(n)
        adj_opinions = []
        for i in adj:
            adj_opinions.append(self.opinions_dict[self.T][i])
        return sum(adj_opinions) / len(adj)

    def __iterate_agent(self, n, upr):
        """Calculate new opinion for agent with index = n"""
        r.seed(100)
        new_opinion = self.opinions_dict[self.T][n] + self.a[n] * \
            (self.mean_for_adj(n) - self.opinions_dict[self.T][n])
        if self.agents_list[n].under_control:
            new_opinion += upr #+ r.uniform(-0.09, 0.09)
        self.agents_list[n].opinion = new_opinion + normal(0, pow(0.3, 2))
        return new_opinion

    def iterate_model(self, upr):
        """Iterate model and fill new data about opinion in agents_list and opinions_dict"""
        for i in range(0, len(self.adj_matrix)):
            self.__iterate_agent(i, upr)
        self.T += 1
        self.opinions_dict[self.T] = [i.opinion for i in self.agents_list]
        self.is_stopped(upr)

    def is_sufficient_delta(self):
        """Return False if model has no changes between last 2 iterations"""
        last = self.opinions_dict[self.T]
        prev = self.opinions_dict[self.T - 1]
        delta = [abs(last[i] - prev[i]) for i in range(0, len(self.agents_list))]
        if max(delta) < 0.0001:
            return False
        return True

    def is_stopped(self, upr):
        """Check if model stopped evolve"""
        if max([abs(i.opinion - self.x_hat) for i in self.agents_list]) < 0.02:
            self.stopped = True
            self.exitcode = 1
            print(self.exitcode)
            return
        if self.T > 160:
            self.stopped = True
            self.exitcode = 2
            print(self.exitcode)
            return
        if max(self.opinions_dict[self.T]) > 1.5:
            self.stopped = True
            self.exitcode = 3
            print(self.exitcode)
            return
        if upr == 0:
            self.stopped = True
            self.exitcode = 4
            print(self.exitcode)
            return
        if not self.is_sufficient_delta():
            if max([abs(i - self.x_hat) for i in self.opinions_dict[self.T]]) < self.err:
                self.stopped = True
                self.exitcode = 1
                print(self.exitcode)
                return
            self.stopped = True
            self.exitcode = 5
            print(self.exitcode)
            return
