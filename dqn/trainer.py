from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import threading
import signal

from util.imgutil import process_image
from .agent import Agent


class Trainer(object):

    def __init__(self, config):
        # env = gym.make('Enduro-v0')
        self.env = gym.make(config.env_name)
        config.checkpoint_dir = config.save_dir + '/checkpoints'
        config.log_dir = config.save_dir + '/logs'
        config.action_dim = self.env.action_space.n

        self.config = config
        self.agent = Agent(config)

        self.stop_requested = False
        return

    def _train_function(self):
        # from PIL import Image
        config = self.config
        o_t = self.env.reset()
        o_t = process_image(o_t, (110, 84), (0, 20, config.state_dim, 20 + config.state_dim), config.use_rgb)
        s_t = np.concatenate([o_t, o_t, o_t, o_t], axis=2)
        while not self.stop_requested and self.agent.global_t < self.config.max_time_step:
            self.env.render()
            action, action_q = self.agent.pick_action(s_t, reward=0.0, use_epsilon_greedy=True)
            o_t1, reward, done, info = self.env.step(action)

            o_t1 = process_image(o_t1, (110, 84), (0, 20, config.state_dim, 20 + config.state_dim), config.use_rgb)
            # Image.fromarray(np.reshape(o_t1, [84, 84])).save('tmp/%d.png' % (self.agent.global_t))
            s_t1 = np.concatenate([s_t[:, :, 3 if config.use_rgb else 1:], o_t1], axis=2)

            if done:
                o_t1 = self.env.reset()
                o_t1 = process_image(o_t1, (110, 84), (0, 20, config.state_dim, 20 + config.state_dim), config.use_rgb)
                s_t1 = np.concatenate([o_t1, o_t1, o_t1, o_t1], axis=2)

            self.agent.perceive(s_t, action, reward, s_t1, done)
            s_t = s_t1
            if self.agent.global_t % 100 == 0 or reward > 0.0:
                print ('global_t=%d / action_id=%d reward=%.2f / epsilon=%.6f / Q=%.4f'
                       % (self.agent.global_t, action, reward, self.agent.epsilon, action_q))
        return

    def signal_handler(self, signal_, frame_):
        print('You pressed Ctrl+C !')
        self.stop_requested = True
        return

    def run(self):
        train_thread = threading.Thread(target=self._train_function)

        signal.signal(signal.SIGINT, self.signal_handler)
        train_thread.start()

        print('Press Ctrl+C to stop')
        signal.pause()
        print('Now saving data....')
        train_thread.join()
        self.agent.backup_session()
        return
