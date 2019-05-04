import gym
import numpy as np


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
    print(state_dim, action_dim, action_bound)
    done = True
    
    while True:
        if done:
            observation = env.reset()
        env.render()
        action = (np.random.rand() - 0.5) * 2
        action = 
        observation, reward, done, info = env.step([action])


        
