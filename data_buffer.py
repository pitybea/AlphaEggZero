# CopyRight no@none.not
import numpy as np


def format_arr(arr):
    return '[' + ', '.join(['%f' % a for a in arr]) + ']'

class AlphaDataBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.win_lose_buffer = []
        self.action_posibility_buffer = []

    def __str__(self):
        return 'buffer size: %d\n' % len(self.data_buffer) + '\n'.join(['%d, %d, ' %(np.argmax(self.data_buffer[i]) + 1, self.win_lose_buffer[i][0]) + format_arr(self.action_posibility_buffer[i]) for i in range(len(self.data_buffer))]) 
        
    def add_one_data(self, data, win_lose, action_posibility):
        self.data_buffer.append(data)
        self.win_lose_buffer.append([win_lose])
        self.action_posibility_buffer.append(action_posibility)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            self.win_lose_buffer.pop(0)
            self.action_posibility_buffer.pop(0)

    def get_data(self):
        indx = np.random.permutation(len(self.data_buffer))
        return np.array(self.data_buffer)[indx], [np.array(self.win_lose_buffer)[indx],
                                                  np.array(self.action_posibility_buffer)[indx]]


class DDPGDataBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.state_buffer = []
        self.q_buffer = []
        
    def add_one_data(self, state, q):
        self.sate_buffer.append(state)
        self.q_buffer.append([q])

        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
            self.q_buffer.pop(0)

    def get_batch(self, size):
        indx = np.random.permutation(len(self.data_buffer))[ : size]
        return np.array(self.data_buffer)[indx], np.array(self.win_lose_buffer)[indx]
