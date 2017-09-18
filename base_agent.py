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

    def restore_session(self):
        checkpoint = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print ('checkpoint loaded:', checkpoint.model_checkpoint_path)
            tokens = checkpoint.model_checkpoint_path.split('-')
            # set global step
            self.global_t = int(tokens[1])
            print ('>>> global step set: ', self.global_t)
        else:
            print ('Could not find old checkpoint')
            self.global_t = 0
        return

    def backup_session(self):
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        self.saver.save(self.sess, self.config.checkpoint_dir + '/' + 'checkpoint', global_step=self.global_t)
        return
