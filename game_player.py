#CopyRight no@none.not
from data_buffer import DataBuffer
from egg_game import EggGame, EggGameNode
from ml_model import TwoHeadModel
import numpy as np
import pandas as pd
import json



def play_egg_game(egg_total, max_egg_per_round, buffer_size,
              round_then_train, total_train_times,
              mcts_search_times, mcts_search_depth):
    win_lose_gt = [-1 if i % (max_egg_per_round + 1) == 0  else 1 for i in range(1, egg_total + 1)]
    egg_game = EggGame(egg_total, max_egg_per_round)
    two_head_model = TwoHeadModel(egg_total, max_egg_per_round)
    data_buffer = DataBuffer(buffer_size)

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
        two_head_model.train_model(data_buffer)
        ab = two_head_model.get_status()
        judge = ab[1]
        print(ab[0])
        print(judge)
        print(win_lose_gt)
        ddf = pd.DataFrame()
        ddf['True'] = win_lose_gt
        ddf['Learned'] = judge
        ddf['baseline'] = 0
        ddf.index = range(1, len(judge) + 1)
        ax = ddf.plot()
        ax.get_figure().savefig('output%03d.png' % train_time)

if __name__ == '__main__':
    play_egg_game(egg_total = 9, max_egg_per_round = 2, buffer_size = 15, round_then_train = 30, total_train_times = 50, mcts_search_times = 10, mcts_search_depth = 2)
    
