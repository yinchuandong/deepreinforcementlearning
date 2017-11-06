from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util.network_util import *
from base.base_network import BaseNetwork


class Network(BaseNetwork):

    def __init__(self, input_shape, action_dim, entropy_beta, scope="net", device="/gpu:0"):
        BaseNetwork.__init__(self, scope, device)
        self._input_shape = input_shape
        self._action_dim = action_dim
        self._entropy_beta = entropy_beta
        self._scope = scope
        self._device = device
        self._create_network()
        self._prepare_loss()
        return

    def _create_network(self):
        with tf.device(self._device), tf.variable_scope(self._scope):
            self.states = tf.placeholder(tf.float32, shape=[None] + self._input_shape, name="states")
            self.dropout = tf.placeholder(tf.float32, shape=[], name="dropout")

            W_fc1, b_fc1 = fc_variable([self._input_shape[-1], 10], name="fc1")
            h_fc1 = tf.nn.tanh(tf.matmul(self.states, W_fc1) + b_fc1)

            h_fc1_dropout = tf.nn.dropout(h_fc1, self.dropout)

            W_fc2, b_fc2 = fc_variable([10, self._action_dim], name="fc2")
            h_fc2 = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

            self.logits = h_fc2
            self.pi = tf.nn.softmax(self.logits)
        return

    def _prepare_loss(self):
        with tf.name_scope(self._scope):
            self.actions = tf.placeholder(tf.int32, shape=[None], name="actions")
            self.returns = tf.placeholder(tf.float32, shape=[None], name="returns")

            action_onehot = tf.one_hot(self.actions, self._action_dim)
            # _loss = self.returns * tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=action_onehot)
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
            _loss = tf.reduce_sum(log_pi * action_onehot, axis=1) * self.returns + self._entropy_beta * entropy
            self.loss = -tf.reduce_mean(_loss)
        return
