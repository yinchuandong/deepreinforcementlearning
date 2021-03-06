from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util.network_util import conv_variable, fc_variable
from util.network_util import max_pool_2x2, conv2d, flatten_conv_layer
from base.base_network import BaseNetwork


class Network(BaseNetwork):

    def __init__(self, input_shape, action_dim, use_huber_loss=True, scope="net", device="/gpu:0"):
        BaseNetwork.__init__(self, scope, device)
        self._input_shape = input_shape
        self._action_dim = action_dim
        self._use_huber_loss = use_huber_loss
        self._scope = scope
        self._device = device

        self._create_network()
        self._prepare_loss()
        return

    def _create_network(self):
        with tf.device(self._device), tf.variable_scope(self._scope):
            self.states = tf.placeholder(tf.float32, shape=[None] + self._input_shape, name="states")
            self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

            # state_dropout = tf.nn.dropout(self.states, self.dropout)
            W_conv1, b_conv1 = conv_variable([8, 8, self._input_shape[2], 16], name="conv1")
            h_conv1 = tf.nn.relu(conv2d(self.states, W_conv1, 4) + b_conv1)
            # h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(self.states, W_conv1, 4) + b_conv1))

            # h_pool1 = max_pool_2x2(h_conv1)

            W_conv2, b_conv2 = conv_variable([4, 4, 16, 32], name="conv2")
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
            # h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_pool1, W_conv2, 2) + b_conv2))

            # W_conv3, b_conv3 = conv_variable([3, 3, 64, 64], name="conv3")
            # h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
            # h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, W_conv3, 1) + b_conv3))

            h_conv_flat_size, h_conv_flat = flatten_conv_layer(h_conv2)

            W_fc1, b_fc1 = fc_variable([h_conv_flat_size, 256], name="fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

            h_fc1_dropout = tf.nn.dropout(h_fc1, self.dropout)

            W_fc2, b_fc2 = fc_variable([256, self._action_dim], name="fc2")
            h_fc2 = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

            self.Q = h_fc2
            self.Q_a = tf.argmax(self.Q, axis=1)
        return

    def _prepare_loss(self):
        with tf.name_scope(self._scope):
            self.actions = tf.placeholder(tf.int64, shape=[None], name="actions")
            self.Q_target = tf.placeholder(tf.float32, shape=[None], name="Q_target")

            action_onehot = tf.one_hot(self.actions, self._action_dim)
            Q_value = tf.reduce_sum(self.Q * action_onehot, axis=1)

            diff = self.Q_target - Q_value

            if self._use_huber_loss:
                # huber loss: https://blog.openai.com/openai-baselines-dqn/
                _loss = tf.where(tf.abs(diff) < 1.0, 0.5 * tf.square(diff), tf.abs(diff) - 0.5)
                self.loss = tf.reduce_mean(_loss)
                print("use: huber loss")
            else:
                # self.loss = tf.reduce_mean(tf.abs(diff))
                self.loss = tf.reduce_mean(tf.square(diff))
                print("use: mse loss")
        return
