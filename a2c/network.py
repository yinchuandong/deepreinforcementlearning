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
            self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

            W_conv1, b_conv1 = conv_variable([8, 8, self._input_shape[2], 16], name="conv1")
            h_conv1 = tf.nn.relu(conv2d(self.states, W_conv1, 4) + b_conv1)

            W_conv2, b_conv2 = conv_variable([4, 4, 16, 32], name="conv2")
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

            h_conv_flat_size, h_conv_flat = flatten_conv_layer(h_conv2)

            W_fc1, b_fc1 = fc_variable([h_conv_flat_size, 256], name="fc1")
            h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)

            h_fc1_dropout = tf.nn.dropout(h_fc1, self.dropout)

            W_pi, b_pi = fc_variable([256, self._action_dim], name="pi")
            self.pi = tf.softmax(tf.matmul(h_fc1_dropout, W_pi) + b_pi)

            W_v, b_v = fc_variable([256, 1], name="v")
            self.v = tf.matmul(h_fc1_dropout, W_v) + b_v
        return

    def _prepare_loss(self):
        with tf.name_scope(self._scope):
            self.actions = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.returns = tf.placeholder(tf.float32, shape=[None], name="returns")

            adv = self.returns - self.v
            action_onehot = tf.one_hot(self.actions, self._action_dim)

            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            pi_loss = tf.reduce_sum(log_pi * action_onehot, axis=1) * adv
            pi_loss = -tf.reduce_sum(pi_loss)

            v_loss = 0.5 * tf.nn.l2_loss(adv)
            self.loss = pi_loss + v_loss
        return
