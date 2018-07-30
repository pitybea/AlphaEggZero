#CopyRight no@none.not
from egg_game import EggGame, EggGameNode
from ml_model import TwoHeadModel
import numpy as np
import json


class DataBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.win_lose_buffer = []
        self.action_posibility_buffer = []

    def add_one_data(self, data, win_lose, action_posibility):
        self.data_buffer.append(data)
        self.win_lose_buffer.append([win_lose])
        self.action_posibility_buffer.append(action_posibility)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            self.win_lose_buffer.pop(0)
            self.action_posibility_buffer.pop(0)

    def get_data(self):
        return np.array(self.data_buffer), [np.array(self.win_lose_buffer),
                                            np.array(self.action_posibility_buffer)]


def play_egg_game(egg_total, max_egg_per_round, buffer_size,
              round_then_train, total_train_times,
              mcts_search_times, mcts_search_depth):
    egg_game = EggGame(egg_total, max_egg_per_round)
    two_head_model = TwoHeadModel(egg_total, max_egg_per_round)
    data_buffer = DataBuffer(buffer_size)

    for _ in range(total_train_times):
        for _ in range(round_then_train):
            game_node = EggGameNode(egg_total, 1)
            game_step_count = 0
            while game_node.egg_leftover != 0:
                while game_node.n_visits < mcts_search_times:
                    round_node = game_node
                    step = 0
                    while round_node.egg_leftover != 0 and step <= mcts_search_depth:
                        round_node.expand(egg_game)
                        round_node = round_node.foward_select_PUCT(two_head_model.get_action_posibility(round_node.egg_leftover, round_node.player_label))
                        step += 1
                    gain_new = -1.0 if round_node.egg_leftover == 0 else two_head_model.get_win_lose(round_node.egg_leftover, round_node.player_label)
                    while round_node != None:
                        round_node = round_node.backward_update(gain_new)
                        gain_new *= -1
                    
                game_node = game_node.select_next()
                game_node.step = 0
                
            win_lose = -1.0 * game_node.player_label
            game_node = game_node.parent
            while game_node != None:
                data = np.zeros(egg_total + 1)
                data[-1] = game_node.player_label
                data[game_node.egg_leftover - 1] = 1.0
                action_posibility = [game_node.play_prob[i + 1]
                                     if i + 1 in game_node.play_prob else 0.0
                                     for i in range(max_egg_per_round)]
                data_buffer.add_one_data(data, win_lose, action_posibility)
                game_node = game_node.parent
        print(data_buffer.get_data())
        two_head_model.train_model(data_buffer.get_data())
        print(two_head_model.get_status())
        raw_input()

if __name__ == '__main__':
    play_egg_game(egg_total = 5, max_egg_per_round = 2, buffer_size = 300, round_then_train = 400, total_train_times = 50, mcts_search_times = 15, mcts_search_depth = 2)
