import numpy as np

num_actions = 2
nodeMap = {}

class cfrNode:
    def __init__(self):
        self.regretSum = np.zeros([num_actions])
        self.strategySum = np.zeros([num_actions])
        
    def getStrategy(self):
        ind = self.regretSum > 0
        return (lambda nSum: np.ones([num_actions]) / num_actions if not nSum > 0 else np.maximum(self.regretSum, 0.0)/nSum)(np.sum(self.regretSum[ind]))
    
    def getAverageStrategy(self):
        return self.strategySum/np.sum(self.strategySum)
    
    def update(self, regret, strategy):
        self.regretSum += regret
        self.strategySum += strategy
    
def cfr(cards, history, p):
    player = len(history) % 2
    isPlayerBetter = 1 if cards[player] > cards[1-player] else -1
    end_states = {'pbb': 2 * isPlayerBetter, 'bb': 2 * isPlayerBetter, 'pp': isPlayerBetter, 'pbp': 1, 'bp': 1}
    if history in end_states: return end_states[history]
    infoSet = str(cards[player]) + history
    if infoSet not in nodeMap: nodeMap[infoSet] = cfrNode()
    strategy = nodeMap[infoSet].getStrategy()
    util = np.array([-cfr(cards, history + ['p', 'b'][a], p * np.array([[strategy[a], 1.0], [1.0, strategy[a]]])[player]) for a in range(num_actions)])
    nodeUtil = np.dot(util, strategy)
    nodeMap[infoSet].update((util - nodeUtil) * p[1-player], strategy * p[player])
    return nodeUtil

for _ in range(50000): cfr(np.random.permutation([1, 2, 3]), '', np.array([1.0, 1.0]))
print('\n'.join([s + ':' + str(nodeMap[s].getAverageStrategy()) for s in nodeMap]))
