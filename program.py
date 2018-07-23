#CopyRight no@none.not

class EggGame():
    def __init__(self, egg_total, max_egg_per_round):
        assert max_egg_per_round > 0

        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round

    def statuses_one_step(self, egg_leftover)->list:
        assert egg_leftover <= max_egg_per_round

        return [egg_leftover - i for i in range(min(max_egg_per_round, egg_leftover), 1, -1)]

class EggGameNode():
    def __init__(self, egg_leftover, parent = None):
        self.parent = parent
        self.avg_gain = 0.0
        self.n_visits = 0
        self.egg_leftover = egg_leftover
        self.children = []
    
    def node_status(self, n_rounds):
        if self.egg_leftover == 0:
            return -1 ** (n_rounds + 1)
        return 0

    def expand(self, game):
        if self.children == []:
            statuses_new = game.statuses_one_step(self.egg_leftover)
            self.children = [EggGameNode(s, self) for s in statuses_new]

    def foward_select(self, posibilities):
        pass

    def backward_update(self, gain_new):
        pass
