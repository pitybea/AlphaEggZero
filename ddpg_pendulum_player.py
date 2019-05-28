import gym
import numpy as np
from ddpg_noise import OrnsteinUhlenbeckActionNoise
from ddpg_model import DDPGModel
from ddpg_buffer import DDPGBuffer

def normalize_state(s):
    # the state is 3-dimentional
    # s[0], s1[1] from -1 to 1, s[2] from -8 to 8
    s[2] /= 8.0
    return s

def normalize_reward(r):
    # the reward is in [-16.27..., 0]
    min_reward = -16.2736045
    normal_para = np.abs(min_reward) / 2.0
    return r / normal_para + 1.0

def actual_action(a):
    # the action sapce is [-2, 2]
    return 2.0 * a

if __name__ == '__main__':
    
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(state_dim, action_dim)
    done = True

    noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(action_dim))
    model = DDPGModel(state_dim, action_dim)
    buf = DDPGBuffer(1e6)
    
    while True:
        if done:
            print('one round finished')
            observation = env.reset()
        env.render()
        action = noise()
        print(action)
        observation, reward, done, info = env.step([action])


        
