from ddpg_model import DDPGModel
import numpy as np

net = DDPGModel(3, 1)

for i in range(10):
    net.train_model(np.array([[0.5, 0.3, 0.2]]), np.array([0.2]))
    print(net.get_critic(np.array([[0.5, 0.3, 0.2]]), np.array([[0.2]])))
    print(net.get_action(np.array([[0.5, 0.3, 0.2]]), np.array([[0.2]])))



