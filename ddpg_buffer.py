import numpy as np

class DDPGBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.state_buffer = []
        self.q_buffer = []
        
    def add_one_data(self, state, q):
        self.state_buffer.append(state)
        self.q_buffer.append([q])

        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
            self.q_buffer.pop(0)

    def add_batch(self, states, qs):
        assert(len(states) == len(qs))
        for s, q in zip(states, qs):
            self.add_one_data(s, q)
            
    def get_batch(self, size):
        if len(self.state_buffer) == 0:
            return None, None
        indx = np.random.permutation(len(self.state_buffer))[ : size]
        return np.array(self.state_buffer)[indx], np.array(self.q_buffer)[indx]
