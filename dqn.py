from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import threading
import signal
import random
from collections import deque

from netutil import *
from imgutil import *
from base_network import BaseNetwork
from base_agent import BaseAgent


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
        with tf.device(self._device), tf.variable_scope(self._scope):
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
            self.Q_a = tf.argmax(self.Q, axis=1)
        return

    def _prepare_loss(self):
        with tf.name_scope(self._scope):
            self.action = tf.placeholder(tf.int64, shape=[None], name='action')
            self.Q_target = tf.placeholder(tf.float32, shape=[None], name='Q_target')

            action_onehot = tf.one_hot(self.action, self._action_dim)
            Q_value = tf.reduce_sum(self.Q * action_onehot, axis=1)
            self.loss = tf.reduce_mean(tf.square(self.Q_target - Q_value))
        return

    @property
    def vars(self):
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope)
        return trainable_vars


class DQNAgent(BaseAgent):

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

    def pickAction(self, state, reward, use_epsilon_greedy=True):
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


class DQNTrainer(object):

    def __init__(self, config):
        # env = gym.make('Enduro-v0')
        self.env = gym.make(config.env_name)
        config.checkpoint_dir = config.save_dir + '/checkpoints'
        config.log_dir = config.save_dir + '/logs'
        config.action_dim = self.env.action_space.n

        self.config = config
        self.agent = DQNAgent(config)

        self.stop_requested = False
        return

    def _train_function(self):
        # from PIL import Image
        config = self.config
        o_t = self.env.reset()
        o_t = scale_image(o_t, (config.state_dim, config.state_dim), config.use_rgb)
        s_t = np.concatenate([o_t, o_t, o_t, o_t], axis=2)
        while not self.stop_requested and self.agent.global_t < self.config.max_time_step:
            self.env.render()
            action, action_q = self.agent.pickAction(s_t, reward=0.0, use_epsilon_greedy=True)
            o_t1, reward, done, info = self.env.step(action)
            o_t1 = scale_image(o_t1, (config.state_dim, config.state_dim), config.use_rgb)
            # Image.fromarray(np.reshape(o_t1, [84, 84])).save('tmp/%d.png' % (self.agent.global_t))
            s_t1 = np.concatenate([s_t[:, :, 3 if config.use_rgb else 1:], o_t1], axis=2)
            if done:
                o_t1 = self.env.reset()
                o_t1 = scale_image(o_t1, (config.state_dim, config.state_dim), config.use_rgb)
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


def main(args):
    config = tf.app.flags.FLAGS
    trainer = DQNTrainer(config)
    # trainer._train_function()
    trainer.run()
    return


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('env_name', 'Breakout-v0', 'the name of game to be trained')
    tf.app.flags.DEFINE_string('save_dir', 'tmp_dqn', 'save models and logs')
    tf.app.flags.DEFINE_boolean('use_gpu', False, 'use gpu or cpu to train')
    tf.app.flags.DEFINE_integer('max_time_step', 10 * 10 ** 7, 'max steps to train')
    tf.app.flags.DEFINE_integer('replay_size', 1 * 10 ** 5, 'the size of replay buffer')

    tf.app.flags.DEFINE_boolean('use_double_dqn', False, 'whether use target net to estimate Q_target')
    tf.app.flags.DEFINE_boolean('use_duel_dqn', False, 'whether use duelling channel')
    tf.app.flags.DEFINE_boolean('use_rgb', False, 'whether use rgb or gray image')
    tf.app.flags.DEFINE_integer('state_dim', 84, 'the width and height of state')
    tf.app.flags.DEFINE_integer('state_chn', 4, 'the channel of state')
    # tf.app.flags.DEFINE_integer('action_dim', 5, 'the action size of game')

    tf.app.flags.DEFINE_integer('epsilon_timestep', 1 * 10 ** 5, 'the step of epsilon greedy')
    tf.app.flags.DEFINE_float('epsilon_hi', 1.0, 'maximum epsilon greedy')
    tf.app.flags.DEFINE_float('epsilon_lo', 0.1, 'minimum epsilon greedy')
    tf.app.flags.DEFINE_integer('batch_size', 32, 'batch_size')

    tf.app.flags.DEFINE_float('gamma', 0.99, 'the discounted factor of reward')
    tf.app.flags.DEFINE_float('alpha', 1e-4, 'learning rate')
    tf.app.flags.DEFINE_float('max_gradient', 10.0, 'maximum gradient when clipping gradients')
    tf.app.run()
