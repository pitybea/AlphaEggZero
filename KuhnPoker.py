import numpy as np

num_actions = 2

class cfrNode:

    def __init__(self):
        self.regretSum = np.zeros([num_actions])
        self.strategySum = np.zeros([num_actions])
    
    def getStrategy(self):
        ind = self.regretSum > 0
        normalizedSum = np.sum(self.regretSum[ind])
        if not normalizedSum > 0:
            return np.ones([num_actions])/num_actions
        strategy = np.zeros([num_actions])
        strategy[ind] = self.regretSum[ind]/normalizedSum
        return strategy

    def getAverageStrategy(self):
        return self.strategySum/np.sum(self.strategySum)

    def update(self, regret, strategy):
        self.regretSum += regret
        self.strategySum += strategy
    
nodeMap = {}

def cfr(cards, history, p0, p1):
    player = len(history) % 2
    isPlayerBetter = (cards[player] > cards[1-player])
    
    if history in ['pbb', 'bb']:
        return 2 if isPlayerBetter else -2
    elif history in ['pp']:
        return 1 if isPlayerBetter else -1
    elif history in ['pbp', 'bp']:
        return 1

    infoSet = str(cards[player]) + history
    if infoSet not in nodeMap:
        nodeMap[infoSet] = cfrNode()
    
    strategy = nodeMap[infoSet].getStrategy()
    util = np.array([-cfr(cards, history + ['p', 'b'][a], p0 * strategy[a], p1) if player == 0 else -cfr(cards, history + ['p', 'b'][a], p0, p1 * strategy[a]) for a in range(num_actions)])
    
    nodeUtil = np.sum(util * strategy)
    nodeMap[infoSet].update((util - nodeUtil) * (p1 if player == 0 else p0), strategy * (p0 if player == 0 else p1))

    return nodeUtil

for _ in range(50000):
    cfr(np.random.permutation([1, 2, 3]), '', 1.0, 1.0)

for s in nodeMap:
    print(s, nodeMap[s].getAverageStrategy())
