import numpy as np

num_actions = 2

class CFRNode:
    def __init__(self):
        self.regretSum = np.zeros([num_actions])
        self.strategySum = np.zeros([num_actions])

    def getStrategy(self):
        if not np.any(np.greater(self.regretSum, 0.0)):
            return np.ones([num_actions]) / num_actions
        return  self.regretSum / np.sum(self.regretSum)

    def getAverageStrategy(self):
        return self.strategySum / np.sum(self.strategySum)

    def upgradeRegStrategy(self, reg, strategy):
        indx = np.greater(reg, 0.0)
        self.regretSum[indx] += reg[indx]
        self.strategySum += strategy
    
nodeMap = {}

def cfr(history, i, p0, p1, cards):
    if history in ['pbb', 'bb']:
        return [2, -2][i] if cards[0] > cards[1] else [-2, 2][i]
    elif history == 'pp':
        return [1, -1][i] if cards[0] > cards[1] else [-1, 1][i]
    elif history == 'pbp':
        return [-1, 1][i]
    elif history == 'bp':
        return [1, -1][i]

    infoSet = str(cards[i]) + history
    if infoSet not in nodeMap:
        nodeMap[infoSet] = CFRNode()
    node = nodeMap[infoSet]
    v_sigma = 0.0
    v_sigma_i_a = np.zeros([num_actions])

    strategy = node.getStrategy()
    for a in range(num_actions):
        if len(history) % 2 == 0:
            v_sigma_i_a[a] = cfr(history + ['p', 'b'][a], i, strategy[a] * p0, p1, cards)
        else:
            v_sigma_i_a[a] = cfr(history + ['p', 'b'][a], i, p0, strategy[a] * p1, cards)
        v_sigma += strategy[a] * v_sigma_i_a[a]

    if len(history) % 2 == i:
        node.upgradeRegStrategy((v_sigma_i_a - v_sigma) * (p1 if i == 0 else p1), strategy * (p0 if i == 0 else p1))
    return v_sigma

for t in range(10000):
    for i in [0, 1]:
        cards = np.random.permutation([1, 2, 3])
        cfr('', i, 1.0, 1.0, cards)

for t in nodeMap:
    print(t, nodeMap[t].getAverageStrategy())
        
            
