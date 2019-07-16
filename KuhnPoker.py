import numpy as np

Pass = 0
Bet = 1
num_actions = 2

class kuhnNode:

    def __init__(self, s):
        self.infoSet = s
        self.regretSum = [0.0] * num_actions
        self.strategySum = [0.0] * num_actions
    
    def getStrategy(self, weight):
        strategy = [0.0] * num_actions
        normalizedSum = 0.0
        for a in range(num_actions):
            strategy[a] = (self.regretSum[a] if self.regretSum[a] > 0.0 else 0.0)
            normalizedSum += strategy[a]

        for a in range(num_actions):
            strategy[a] = (strategy[a] / normalizedSum if normalizedSum > 0.0 else 1.0 / num_actions)
            self.strategySum[a] += strategy[a] * weight

        return strategy

    def getAction(self, strategy):
        r = np.random.rand()
        cul_prob = 0.0
        for a in range(num_actions):
            cul_prob += strategy[a]
            if cul_prob > r:
                return a
        return num_actions - 1

    def getAverageStrategy(self):
        normalizedSum = sum(self.strategySum)
        return [t / normalizedSum for t in self.strategySum]

nodeMap = {}

def cfr(cards, history, p0, p1):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    isPlayerBetter = (cards[player] > cards[opponent])
    
    if history in ['pbb', 'bb']:
        return 2 if isPlayerBetter else -2
    elif history in ['pp']:
        return 1 if isPlayerBetter else -1
    elif history in ['pbp']:
        return 1
    elif history in ['bp']:
        return 1

    infoSet = str(cards[player]) + history
    if infoSet not in nodeMap:
        nodeMap[infoSet] = kuhnNode(infoSet)
    
    strategy = nodeMap[infoSet].getStrategy(p0 if player == 0 else p1)
    util = [0.0] * num_actions
    nodeUtil = 0.0
    
    for a in range(num_actions):
        nextHistory = history + ('p' if a == 0 else 'b')
        util[a] = -cfr(cards, nextHistory, p0 * strategy[a], p1) if player == 0 else -cfr(cards, nextHistory, p0, p1 * strategy[a])
        nodeUtil += strategy[a] * util[a]
        
    for a in range(num_actions):
        regret = util[a] - nodeUtil
        nodeMap[infoSet].regretSum[a] += regret * (p1 if player == 0 else p0)

    return nodeUtil

def train(iter_times):
    cards = [1, 2, 3]
    util = 0.0
    for _ in range(iter_times):
        cur_cards = np.random.permutation(cards)
        util += cfr(cur_cards, '', 1.0, 1.0)
    print iter_times, util / iter_times
    for s in nodeMap:
        print s, nodeMap[s].getAverageStrategy()
train(10000)
