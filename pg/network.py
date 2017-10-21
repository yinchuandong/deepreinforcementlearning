from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util.network_util import *
from base.base_network import BaseNetwork


class Network(BaseNetwork):

    def __init__(self, input_shape, action_dim, scope="net", device="/gpu:0"):
        BaseNetwork.__init__(self, scope, device)
        self._input_shape = input_shape
        self._action_dim = action_dim
        self._scope = scope
        self._device = device
        self._create_network()
        self._prepare_loss()
        return

    def _create_network(self):
        with tf.device(self._device), tf.variable_scope(self._scope):
            self.states = tf.placeholder(tf.float32, shape=[None] + self._input_shape, name="states")
            h_scale_states = self.states / 255.0
            self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

            # state_dropout = tf.nn.dropout(self.states, self.dropout)
            W_conv1, b_conv1 = conv_variable([8, 8, self._input_shape[2], 32], name="conv1")
            h_conv1 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_scale_states, W_conv1, 4) + b_conv1))

            W_conv2, b_conv2 = conv_variable([4, 4, 32, 64], name="conv2")
            h_conv2 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv1, W_conv2, 2) + b_conv2))

            W_conv3, b_conv3 = conv_variable([3, 3, 64, 64], name="conv3")
            h_conv3 = tf.nn.relu(tf.layers.batch_normalization(conv2d(h_conv2, W_conv3, 1) + b_conv3))

            h_conv3_flat_size, h_conv3_flat = flatten_conv_layer(h_conv3)

            W_fc1, b_fc1 = fc_variable([h_conv3_flat_size, 512], name="fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, self.dropout)

            W_fc2, b_fc2 = fc_variable([512, self._action_dim], name="fc2")
            h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)
            self.pi = h_fc2
        return

    def _prepare_loss(self):
        with tf.name_scope(self._scope):
            self.actions = tf.placeholder(tf.int64, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")

            action_onehot = tf.one_hot(self.actions, self._action_dim)
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-10, 1.0))
            _loss = -tf.reduce_sum(log_pi * action_onehot, axis=1) * self.rewards
            self.loss = tf.reduce_mean(_loss)
        return
