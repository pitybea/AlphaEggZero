import gym
import numpy as np
from ddpg_noise import OrnsteinUhlenbeckActionNoise
from ddpg_model import DDPGModel
from ddpg_buffer import DDPGBuffer

def normalize_state(s):
    # the state is 3-dimentional
    # s[0], s1[1] from -1 to 1, s[2] from -8 to 8
    s = s.flatten()
    s[2] /= 8.0
    return s

def normalize_reward(r):
    # the reward is in [-16.27..., 0]
    r = r.flatten()[0]
    min_reward = -16.2736045
    normal_para = np.abs(min_reward) / 2.0
    return r / normal_para + 1.0

def actual_action(a):
    # the action sapce is [-2, 2]
    return 2.0 * a

def gamma_normalize_rewards(rs, gamma = 0.9, ep = 5):
    l = len(rs)
    results = [0.0] * l
    weights = [0.0] * l

    for i in range(l - 1, -1, -1):
        w = 1.0
        for j in range(min(i + 1, ep)):
            ind = i - j
            results[ind] += rs[i] * w
            weights[ind] += w
            w = w * gamma
    
    return [results[i] / weights[i] for i in range(l)]

if __name__ == '__main__':
    
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(state_dim, action_dim)
    done = True

    noise = OrnsteinUhlenbeckActionNoise(mu = np.zeros(action_dim))
    model = DDPGModel(state_dim, action_dim)
    buf = DDPGBuffer(1e6)

    ep_state = []
    ep_reward = []

    play_round = 0
    while True:
        if done:
            play_round = play_round + 1
            print('round %d finished' % play_round)
            buf.add_batch(ep_state, gamma_normalize_rewards(ep_reward))
            train_states, train_qs = buf.get_batch(500)
            if train_states is not None:
                model.train_model(train_states, train_qs)
            observation = normalize_state(env.reset())
            print('average reward', np.average(ep_reward))
            ep_state = []
            ep_reward = []
        if play_round % 20 == 0:
            env.render()
        oise = noise()
        action = model.get_action(np.array([observation]), np.array([oise]))[0]
        new_observation, reward, done, info = env.step([actual_action(action)])
        ep_state.append(observation)
        ep_reward.append(normalize_reward(reward))
        observation = normalize_state(new_observation)

        
