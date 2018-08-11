#CopyRight no@none.not
from data_buffer import AlphaDataBuffer
from egg_game import EggGame
from alpha_game_node import EggGameNode
from ml_model import TwoHeadModel
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

def draw_winlose_action_for_model(two_head_model, win_lose_gt, action_gt):
    ab = two_head_model.get_status()
    win_lose_pred = ab[1]
    print(win_lose_gt)
    print(win_lose_pred)
        
    x = range(1, len(win_lose_pred) + 1)
    l1, = plt.plot(x, win_lose_gt, c = 'green')
    l2, = plt.plot(x, win_lose_pred, c = 'red')
    l3, = plt.plot(x, [0] * len(win_lose_pred), c = 'black')

    p1 = plt.scatter(x, np.array(action_gt) + 1, s = 55, c = 'blue')
    p2 = plt.scatter(x, np.array(ab[0]) + 1, s = 15, c = 'yellow')
    
    return [l1, l2, l3, p1, p2]


def play_egg_game(egg_total, max_egg_per_round, buffer_size, round_then_train, total_train_times, mcts_search_times, mcts_search_depth):
    
    win_lose_gt = [-1 if i % (max_egg_per_round + 1) == 0  else 1 for i in range(1, egg_total + 1)]
    action_gt = [i % (max_egg_per_round + 1) if i % (max_egg_per_round + 1) != 0 else np.nan for i in range(1, egg_total + 1)]

    fig = plt.figure()
    plt.xticks(np.arange(1, egg_total + 1))
    plt.yticks(np.arange(-1, max_egg_per_round + 2), ['-1', '0', '1'] + [str(i + 1) for i in range(max_egg_per_round)] )
    
    egg_game = EggGame(egg_total, max_egg_per_round)
    two_head_model = TwoHeadModel(egg_total, max_egg_per_round)
    data_buffer = AlphaDataBuffer(buffer_size)

    ani_buffer = []
    
    for train_time in range(total_train_times):
        for _ in range(round_then_train):
            game_node = EggGameNode(egg_total, 1)
            while game_node.egg_leftover != 0:
                while game_node.n_visits < mcts_search_times:
                    round_node = game_node
                    step = 0
                    while round_node.egg_leftover != 0 and step <= mcts_search_depth:
                        round_node.expand(egg_game)
                        round_node = round_node.foward_select_PUCT(two_head_model.get_action_posibility(round_node.egg_leftover))
                        step += 1
                    gain_new = -1.0 * round_node.player_label if round_node.egg_leftover == 0 else two_head_model.get_win_lose(round_node.egg_leftover) * round_node.player_label
                    while round_node != None:
                        round_node = round_node.backward_update(gain_new)
                    
                game_node = game_node.select_next()
                
            win_lose = -1.0 * game_node.player_label
            game_node = game_node.parent
            while game_node != None:
                data = np.zeros(egg_total)
                data[game_node.egg_leftover - 1] = 1.0
                action_posibility = [game_node.play_prob[i + 1]
                                     if i + 1 in game_node.play_prob else 0.0
                                     for i in range(max_egg_per_round)]
                data_buffer.add_one_data(data, win_lose * game_node.player_label, action_posibility)
                game_node = game_node.parent
        two_head_model.train_model(data_buffer.get_data())
        frame = draw_winlose_action_for_model(two_head_model, win_lose_gt, action_gt)
        ani_buffer.append(frame)
    ArtistAnimation(fig, ani_buffer, interval = 800, blit = True, repeat_delay = 1000).save('learning.mp4', writer = 'ffmpeg')
        
if __name__ == '__main__':
    play_egg_game(egg_total = 20, max_egg_per_round = 4, buffer_size = 25, round_then_train = 10, total_train_times = 100, mcts_search_times = 20, mcts_search_depth = 2)
    
