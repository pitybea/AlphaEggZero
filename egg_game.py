#CopyRight no@none.not
import numpy as np


class EggGame():
    def __init__(self, egg_total, max_egg_per_round):
        assert max_egg_per_round > 0
        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round

    def feasible_actions(self, egg_leftover):
        assert egg_leftover <= self.egg_total
        return [i for i in range(1, min(self.max_egg_per_round, egg_leftover) + 1)]

    
def format_dict(d):
    return '{' + ', '.join('%d: %f' % (a, d[a]) for a in d) + '}'

class EggGameNode():
    def __init__(self, egg_leftover, player_label, parent = None):
        self.parent = parent
        self.player_label = player_label
        self.avg_gain = 0.0
        self.n_visits = 0
        self.egg_leftover = egg_leftover
        self.children = {}
        self.play_prob = {}

    def __str__(self):
        return '{"avg_gain": %f, "n_visits": %d, "prob": %s, "egg_leftover": %d, "player": %d' % (self.avg_gain, self.n_visits, format_dict(self.play_prob), self.egg_leftover, self.player_label) + ', "children": {' + ', '.join(['\n  "%d-c-%d": %s'%(self.egg_leftover, a, self.children[a]) for a in self.children]) + '}}'
        
    def expand(self, game):
        if self.egg_leftover > 0 and self.children == {}:
            actions = game.feasible_actions(self.egg_leftover)
            self.children = {a: EggGameNode(self.egg_leftover - a, self.player_label * -1, self)
                             for a in actions}
            
    def foward_select_PUCT(self, P):
        assert self.children != {}
        N = self.n_visits
        C = 1.0
        scores = {a: np.random.rand() * 0.01 + self.children[a].avg_gain * self.player_label +
                  C * P[a - 1] * np.sqrt(self.n_visits) / (1.0 + self.children[a].n_visits)
                  for a in self.children}
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
        N = sum([self.children[a].n_visits for a in self.children])
        self.play_prob = {a: 1.0 * self.children[a].n_visits / N for a in self.children}    
        choice = np.random.choice(self.play_prob.keys(), 1, p = self.play_prob.values())[0]
        self.children = {choice: self.children[choice]}
        return self.children[choice]
