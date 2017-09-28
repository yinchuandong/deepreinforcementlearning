from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from collections import deque

from util.imgutil import *
from base.base_agent import BaseAgent
from .network import Network


class Agent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self)

        self.config = config
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.sess = tf.Session(config=sess_config)

        # create networks
        state_chn = config.state_chn * (3 if config.use_rgb else 1)
        input_shape = [config.state_dim, config.state_dim, state_chn]
        device = '/gpu:0' if config.use_gpu else '/cpu:0'
        self.main_net = Network(input_shape, config.action_dim, 'main_net', device)
        self.target_net = Network(input_shape, config.action_dim, 'targe_net', device)
        self.sync_target_net = self.target_net.sync_from(self.main_net)

        # create gradients operations
        optimizer = tf.train.RMSPropOptimizer(config.alpha, decay=0.99)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, config.max_gradient) for grad in gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        # summary
        self.merged_summary = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(config.log_dir + '/train', self.sess.graph)

        # initialize parameters
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.restore_session()

        self.epsilon = self._anneal_epsilon(self.global_t)
        self.replay_buffer = deque(maxlen=config.replay_size)
        return

    def _anneal_epsilon(self, timestep):
        config = self.config
        span = float(config.epsilon_hi - config.epsilon_lo) / float(config.epsilon_timestep)
        epsilon = config.epsilon_hi - span * min(timestep, config.epsilon_timestep)
        return epsilon

    def pick_action(self, state, reward, use_epsilon_greedy=True):
        Q_value = self.sess.run(self.main_net.Q, feed_dict={self.main_net.state: [state]})
        Q_value = Q_value[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.config.action_dim)
        else:
            action_index = np.argmax(Q_value)
        max_q_value = np.max(Q_value)
        return action_index, max_q_value

    def perceive(self, state, action, reward, next_state, done):
        self.global_t += 1
        self.epsilon = self._anneal_epsilon(self.global_t)
        self.replay_buffer.append([state, action, reward, next_state, done])
        if len(self.replay_buffer) > self.config.batch_size * 2:
            self._update_weights()

        if self.global_t % 100000 == 0:
            self.backup_session()
        return

    def _update_weights(self):
        minibatch = random.sample(self.replay_buffer, self.config.batch_size)
        batch_state = [t[0] for t in minibatch]
        batch_action = [t[1] for t in minibatch]
        batch_reward = [t[2] for t in minibatch]
        batch_next_state = [t[3] for t in minibatch]
        batch_done = np.array([int(t[4]) for t in minibatch])

        if self.config.use_double_dqn:
            Q_a_next = self.sess.run(self.target_net.Q_a, feed_dict={self.target_net.state: batch_next_state})
            Q_next = self.sess.run(self.main_net.Q, feed_dict={self.main_net.state: batch_next_state})
            double_q = Q_next[range(self.config.batch_size), Q_a_next]
            Q_target = batch_reward + (1.0 - batch_done) * self.config.gamma * double_q
        else:
            Q_next = self.sess.run(self.main_net.Q, feed_dict={self.main_net.state: batch_next_state})
            Q_target = batch_reward + (1.0 - batch_done) * self.config.gamma * np.max(Q_next, axis=1)

        self.sess.run([self.apply_gradients, self.merged_summary], feed_dict={
            self.main_net.state: batch_state,
            self.main_net.action: batch_action,
            self.main_net.Q_target: Q_target,
        })

        if self.config.use_double_dqn:
            self.sess.run(self.sync_target_net)
        return


