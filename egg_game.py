#CopyRight no@none.not
import numpy as np


class EggGame():
    def __init__(self, egg_total, max_egg_per_round):
        assert max_egg_per_round > 0
        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round

    def feasible_actions(self, egg_leftover)->list:
        assert egg_leftover <= max_egg_per_round
        return [i for i in range(1, min(max_egg_per_round, egg_leftover))]

class EggGameNode():
    def __init__(self, egg_leftover, parent = None, step = 0):
        self.parent = parent
        self.avg_gain = 0.0
        self.n_visits = 0
        self.egg_leftover = egg_leftover
        self.children = {}
        self.step = step
        
        self.next_node = None
        self.prev_node = None
    
    def expand(self, game):
        if self.egg_leftover > 0 and self.children == {}:
            actions = game.feasible_actions(self.egg_leftover)
            self.children = {a: EggGameNode(self.egg_leftover - a, self, self.step + 1) for a in actions}

    def foward_select_PUCT(self, P):
        assert self.children != {}
        N = self.n_visits
        C = 1.0
        scores = {a: self.children[a].avg_gain + C * P[a - 1] / (1.0 + self.children[a].n_visits) for a in self.children}
        a = max(scores, key = scores.get)
        return self.children[a]

    def backward_update(self, gain_new):
        assert self != None
        self.avg_gain = (self.avg_gain * self.n_visits + gain_new) / (1.0 + self.n_visits)
        self.n_visits += 1.0
        return self.parent

    def select_next(self):
        assert self.children != {}
        assert self.n_visits > 0
        N = self.n_visits
        P = {a: 1.0 * self.children[a].n_visits / N for a in self.children}
        choice = np.random.choice(P.keys(), 1, p = P.values())[0]
        for a in self.children:
            if a != choice:
                del self.children[a]

        self.next_node = self.children[choice]
        self.children[choice].prev_node = self
        return self.children[choice]
