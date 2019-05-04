from ddpg_model import DDPGModel
import numpy as np

net = DDPGModel(3, 1)

print(net.get_critic(np.array([[0.5, 0.3, 0.2]]), np.array([[0.2]])))
print(net.get_action(np.array([[0.5, 0.3, 0.2]]), np.array([[0.2]])))

net.to_critic_net()
net.describe()

net.to_actor_net()
net.describe()
