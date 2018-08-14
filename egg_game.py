#CopyRight no@none.not

class EggGame():
    def __init__(self, egg_total, max_egg_per_round):
        assert max_egg_per_round > 0
        self.egg_total = egg_total
        self.max_egg_per_round = max_egg_per_round

    def feasible_actions(self, egg_leftover):
        assert egg_leftover > 0
        assert egg_leftover <= self.egg_total
        return [i for i in range(1, min(self.max_egg_per_round, egg_leftover) + 1)]

