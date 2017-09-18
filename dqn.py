from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from netutil import *
from imgutil import *
from base_network import BaseNetwork


class Network(BaseNetwork):

    def __init__(self, input_shape, action_dim, scope, device='/gpu:0'):
        BaseNetwork.__init__(self, device)
        self._input_shape = input_shape
        self._action_dim = action_dim
        self._scope = scope
        self._device = device
        self._create_network()
        self._prepare_loss()
        return

    def _create_network(self):
        with tf.device(self._device), tf.variable_scope(self._scope) as scope:
            self.state = tf.placeholder(tf.float32, shape=[None] + self._input_shape, name='state')

            W_conv1, b_conv1 = conv_variable([8, 8, self._input_shape[2], 16], name='conv1')
            h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1, 4) + b_conv1)

            W_conv2, b_conv2 = conv_variable([4, 4, 16, 32], name='conv2')
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

            h_conv2_flat_size, h_conv2_flat = flatten_conv_layer(h_conv2)

            W_fc1, b_fc1 = fc_variable([h_conv2_flat_size, 64], name='fc1')
            h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

            W_fc2, b_fc2 = fc_variable([64, self._action_dim], name='fc2')
            h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
            self.Q = h_fc2
        return

    def _prepare_loss(self):
        self.action = tf.placeholder(tf.float32, shape=[None, self._action_dim])
        self.Q_target = tf.placeholder(tf.float32, shape=[None])

        Q_value = tf.reduce_sum(self.Q * self.action, axis=1)
        self.loss = tf.reduce_mean(tf.square(self.Q_target - Q_value))
        return

    @property
    def vars(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)
        return trainable_vars


class DQN(object):

    def __init__(self, config):
        # env = gym.make('Enduro-v0')
        self.config = config
        self.env = gym.make('Breakout-v0')
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.sess = tf.Session(config=sess_config)

        action_dim = self.env.action_space.n
        state_chn = config.state_chn * (3 if config.use_rgb else 1)
        input_shape = [config.state_dim, config.state_dim, state_chn]
        device = '/gpu:0' if config.use_gpu else '/cpu:0'

        self.main_net = Network(input_shape, action_dim, 'main_net', device)
        self.target_net = Network(input_shape, action_dim, 'targe_net', device)
        self.sync_target_net = self.target_net.sync_from(self.main_net)

        optimizer = tf.train.RMSPropOptimizer(config.alpha, decay=0.99)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, config.max_gradient) for grad in gradients]

        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        self.sess.run(tf.global_variables_initializer())
        return

    def train(self):

        self.env.reset()
        # for _ in range(1000):
        for _ in range(1):
            self.env.render()
            action = self.env.action_space.sample()
            # print action
            observation, reward, done, info = self.env.step(action)
            im = scale_image(observation, gray=True)
            im.save('1.png')
            if done:
                self.env.reset()
        return


def main(args):
    config = tf.app.flags.FLAGS
    dqn = DQN(config)
    # dqn.train()
    # net = Network([84, 84, 4], 5, 'main', '/cpu:0')
    # print net.vars
    return


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('env_name', 'game', 'name')
    tf.app.flags.DEFINE_string('save_dir', 'tmp_dqn', 'save models and logs')
    tf.app.flags.DEFINE_boolean('use_gpu', False, 'use gpu or cpu to train')
    tf.app.flags.DEFINE_integer('max_time_step', 10 * 10 ** 7, 'max steps to train')
    tf.app.flags.DEFINE_integer('replay_size', 1000000, 'the size of replay buffer')

    tf.app.flags.DEFINE_boolean('use_rgb', True, 'whether use rgb or gray image')
    tf.app.flags.DEFINE_integer('state_dim', 84, 'the width and height of state')
    tf.app.flags.DEFINE_integer('state_chn', 4, 'the channel of state')
    # tf.app.flags.DEFINE_integer('action_dim', 5, 'the action size of game')

    tf.app.flags.DEFINE_integer('epsilon_time_step', 1 * 10 ** 5, 'the step of epsilon greedy')
    tf.app.flags.DEFINE_float('epsilon_hi', 0.8, 'maximum epsilon greedy')
    tf.app.flags.DEFINE_float('epsilon_lo', 0.1, 'minimum epsilon greedy')
    tf.app.flags.DEFINE_integer('batch_size', 128, 'batch_size')

    tf.app.flags.DEFINE_float('gamma', 0.99, 'the discounted factor of reward')
    tf.app.flags.DEFINE_float('alpha', 1e-3, 'learning rate')
    tf.app.flags.DEFINE_float('max_gradient', 10.0, 'maximum gradient when clipping gradients')
    tf.app.run()
