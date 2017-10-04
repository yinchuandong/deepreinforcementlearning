from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


class BaseAgent(object):

    def __init__(self):
        return

    def pickAction(self, state, reward=0.0, use_epsilon_greedy=True):
        raise NotImplementedError('please override step function in BaseAgent')
