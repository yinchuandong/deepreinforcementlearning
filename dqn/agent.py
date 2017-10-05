from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from collections import deque

from base.base_agent import BaseAgent
from util.network_util import restore_session, backup_session
from util.feature_util import process_image
from .network import Network


class Agent(BaseAgent):

    def __init__(self, config):
        BaseAgent.__init__(self)

        self.config = config

        # create networks
        state_chn = config.state_chn * (3 if config.use_rgb else 1)
        input_shape = [config.state_dim, config.state_dim, state_chn]
        device = "/gpu:0" if config.use_gpu else "/cpu:0"
        self.main_net = Network(input_shape, config.action_dim, "main_net", device)
        self.target_net = Network(input_shape, config.action_dim, "target_net", device)
        self.sync_target_net = self.target_net.sync_from(self.main_net)

        # create gradients operations
        optimizer = tf.train.RMSPropOptimizer(config.lr, decay=0.99)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, config.max_gradient) for grad in gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        self.replay_buffer = deque(maxlen=config.replay_size)
        self.stop_requested = False

        # initialize parameters
        # self.add_train_summary(sess)
        # sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver()
        # self.global_t = restore_session(self.saver, sess, config.model_dir)
        # self.epsilon = self._anneal_epsilon(self.global_t)
        return

    def add_train_summary(self, sess):
        tf.summary.scalar("loss", self.main_net.loss)
        self.train_summary = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(self.config.log_dir + "/train", sess.graph)
        return

    def _anneal_epsilon(self, timestep):
        config = self.config
        span = float(config.epsilon_hi - config.epsilon_lo) / float(config.epsilon_timestep)
        epsilon = config.epsilon_hi - span * min(timestep, config.epsilon_timestep)
        return epsilon

    def pick_action(self, sess, state, reward, use_epsilon_greedy=True):
        Q_value = sess.run(self.main_net.Q, feed_dict={self.main_net.state: [state]})
        Q_value = Q_value[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.config.action_dim)
        else:
            action_index = np.argmax(Q_value)
        max_q_value = np.max(Q_value)
        return action_index, max_q_value

    def _perceive(self, sess, state, action, reward, next_state, done):
        self.global_t += 1
        self.epsilon = self._anneal_epsilon(self.global_t)
        self.replay_buffer.append([state, action, reward, next_state, done])
        return

    def _update_weights(self, sess):
        minibatch = random.sample(self.replay_buffer, self.config.batch_size)
        batch_state = [t[0] for t in minibatch]
        batch_action = [t[1] for t in minibatch]
        batch_reward = [t[2] for t in minibatch]
        batch_next_state = [t[3] for t in minibatch]
        batch_done = np.array([int(t[4]) for t in minibatch])

        if self.config.use_double_dqn:
            Q_a_next = sess.run(self.target_net.Q_a, feed_dict={self.target_net.state: batch_next_state})
            Q_next = sess.run(self.main_net.Q, feed_dict={self.main_net.state: batch_next_state})
            double_q = Q_next[range(self.config.batch_size), Q_a_next]
            Q_target = batch_reward + (1.0 - batch_done) * self.config.gamma * double_q
        else:
            Q_next = sess.run(self.main_net.Q, feed_dict={self.main_net.state: batch_next_state})
            Q_target = batch_reward + (1.0 - batch_done) * self.config.gamma * np.max(Q_next, axis=1)

        _, loss, summary = sess.run([self.apply_gradients, self.main_net.loss, self.train_summary], feed_dict={
            self.main_net.state: batch_state,
            self.main_net.action: batch_action,
            self.main_net.Q_target: Q_target,
        })

        if self.global_t % 10 == 0:
            self.train_summary_writer.add_summary(summary, self.global_t)

        if self.config.use_double_dqn and self.global_t % self.config.net_update_step == 0:
            sess.run(self.sync_target_net)
        return

    def train(self, saver, sess, env):
        cfg = self.config
        # summary
        self.add_train_summary(sess)
        self.global_t = restore_session(saver, sess, cfg.model_dir)
        self.epsilon = self._anneal_epsilon(self.global_t)

        while not self.stop_requested and self.global_t < cfg.max_train_step:
            print("-------new epoch-----------------")
            o_t = env.reset()
            o_t = process_image(o_t, (110, 84), (0, 20, cfg.state_dim, 20 + cfg.state_dim), cfg.use_rgb)
            s_t = np.concatenate([o_t, o_t, o_t, o_t], axis=2)
            done = False
            last_action = None

            local_t = 0
            while not done and not self.stop_requested and self.global_t < cfg.max_train_step:
                env.render()
                local_t += 1
                # frame skipping
                if local_t % cfg.frame_skip != 0 and last_action is not None:
                    action_q = -1  # skipping
                    o_t1, reward, done, info = env.step(last_action)
                else:
                    action, action_q = self.pick_action(sess, s_t, reward=0.0, use_epsilon_greedy=True)
                    o_t1, reward, done, info = env.step(action)
                o_t1 = process_image(o_t1, (110, 84), (0, 20, cfg.state_dim, 20 + cfg.state_dim), cfg.use_rgb)
                # Image.fromarray(np.reshape(o_t1, [84, 84])).save("tmp/%d.png" % (self.global_t))
                s_t1 = np.concatenate([s_t[:, :, 3 if cfg.use_rgb else 1:], o_t1], axis=2)

                self._perceive(sess, s_t, action, reward, s_t1, done)
                if len(self.replay_buffer) > self.config.batch_size * 2:
                    self._update_weights(sess)

                s_t = s_t1
                last_action = action
                if self.global_t % 100 == 0 or reward > 0.0:
                    print("global_t=%d / action_id=%d reward=%.2f / epsilon=%.6f / Q=%.4f"
                          % (self.global_t, action, reward, self.epsilon, action_q))

        backup_session(saver, sess, cfg.model_dir, self.global_t)
        return
