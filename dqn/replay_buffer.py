from __future__ import absolute_import

from collections import deque

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer = deque(buffer_size)
        return

    def add(self, img, action, reward, terminal, start_frame):
        self.buffer.add((img, action, reward, terminal, start_frame))
        return


    def get_state_by_idx(self, index):
        o_t1, action, reward, terminal, start_frame = self.buffer[index]

        # for 
        return