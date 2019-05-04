from data_buffer import DDPGDataBuffer
from egg_game import EggGame
from game_node import DDPGEggGameNode
from ddpg_model import DDPGModel
import numpy as np

def play_egg_game(egg_total, max_egg_per_round, buffer_size, round_then_train, total_train_times):
    win_lose_gt = [-1 if i % (max_egg_per_round + 1) == 0  else 1 for i in range(1, egg_total + 1)]
    action_gt = [i % (max_egg_per_round + 1) if i % (max_egg_per_round + 1) != 0 else np.nan for i in range(1, egg_total + 1)]

    egg_game = EggGame(egg_total, max_egg_per_round)
    ddpg_model = DDPGModel(egg_total, max_egg_per_round)
    data_buffer = DDPGDataBuffer(buffer_size)

    for _ in range(total_train_times):
        for _ in range(round_then_train):
            game_node = DDPGEggGameNode(egg_total, 1)
            game_buffer = [game_node]
            while game_node.egg_leftover != 0:
                action = ddpg_model.get_action_posibility(game_node.egg_leftover)
                game_node = game_node.select_next(egg_game, action)
                game_buffer.append(game_node)
                
            win_lose = -1.0 * game_buffer[-1].player_label
            print([(n.egg_leftover, n.action, win_lose * n.player_label) for n in game_buffer])
            for game_node in game_buffer[0: -1]:
                data = np.zeros(egg_total)
                data[game_node.egg_leftover - 1] = 1.0
                data_buffer.add_one_data(data, win_lose * game_node.player_label)
        ddpg_model.train_critic_net(data_buffer.get_data())
        print(ddpg_model.get_status())
        ddpg_model.train_actor_net(data_buffer.get_data())
        print(ddpg_model.get_status())
        print(action_gt, win_lose_gt)

    
if __name__ == '__main__':
    play_egg_game(egg_total = 6, max_egg_per_round = 2, buffer_size = 60, round_then_train = 50, total_train_times = 50)
