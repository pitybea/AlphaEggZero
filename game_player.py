#CopyRight no@none.not
from egg_game import EggGame, EggGameNode
from ml_model import TwoHeadModel
import numpy as np


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
        if len(self.data_buffer) > buffer_size:
            self.data_buffer.pop(0)
            self.win_lose_buffer.pop(0)
            self.action_posibility.pop(0)

    def get_data(self):
        return np.array(self.data_buffer), [np.array(self.win_lose_buffer), np.array(self.action_posibility_buffer)]

