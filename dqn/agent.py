from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from collections import deque
from PIL import Image

from base.base_agent import BaseAgent
from util.network_util import restore_session, backup_session
from util.feature_util import create_process_fn
from .network import Network


class Agent(BaseAgent):

    def __init__(self, cfg, logger):
        BaseAgent.__init__(self)

        self.cfg = cfg
        self.logger = logger

        # create networks
        state_chn = cfg.state_history * (3 if cfg.use_rgb else 1)
        input_shape = [cfg.state_dim, cfg.state_dim, state_chn]
        device = "/gpu:0" if cfg.use_gpu else "/cpu:0"
        self.main_net = Network(input_shape, cfg.action_dim, cfg.use_huber_loss, "main_net", device)
        self.target_net = Network(input_shape, cfg.action_dim, cfg.use_huber_loss, "target_net", device)
        self.sync_target_net = self.target_net.sync_from(self.main_net)

        # create gradients operations
        optimizer = tf.train.RMSPropOptimizer(cfg.lr, decay=cfg.lr_decay)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, cfg.max_grad) for grad in gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        self.replay_buffer = deque(maxlen=cfg.replay_size)
        self.stop_requested = False
        self.global_t = 0
        self.n_episode = 0
        return

    def add_train_summary(self, sess):
        tf.summary.scalar("loss", self.main_net.loss)
        self.train_summary = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(self.cfg.log_dir + "/train", sess.graph)
        return

    def _anneal_epsilon(self, timestep):
        cfg = self.cfg
        span = float(cfg.eps_hi - cfg.eps_lo) / float(cfg.eps_step)
        epsilon = cfg.eps_hi - span * min(timestep, cfg.eps_step)
        return epsilon

    def pick_action(self, sess, state, reward, use_epsilon_greedy=True):
        Q_value = sess.run(self.main_net.Q, feed_dict={
            self.main_net.states: [state], self.main_net.dropout: 1.0
        })
        Q_value = Q_value[0]
        action_index = 0
        if random.random() <= self.epsilon:
            action_index = random.randrange(self.cfg.action_dim)
        else:
            action_index = np.argmax(Q_value)
        max_q_value = np.max(Q_value)
        return action_index, max_q_value

    def _perceive(self, sess, state, action, reward, next_state, done):
        self.global_t += 1
        self.epsilon = self._anneal_epsilon(self.global_t)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return

    def _update_weights(self, sess):
        minibatch = random.sample(self.replay_buffer, self.cfg.batch_size)
        batch_state = [t[0] for t in minibatch]
        batch_action = [t[1] for t in minibatch]
        batch_reward = [t[2] for t in minibatch]
        batch_next_state = [t[3] for t in minibatch]
        batch_done = np.array([int(t[4]) for t in minibatch])

        if self.cfg.use_double_dqn:
            Q_a_next = sess.run(self.target_net.Q_a, feed_dict={
                self.target_net.states: batch_next_state, self.target_net.dropout: 1.0
            })
            Q_next = sess.run(self.main_net.Q, feed_dict={
                self.main_net.states: batch_next_state, self.main_net.dropout: 1.0
            })
            double_q = Q_next[range(self.cfg.batch_size), Q_a_next]
            Q_target = batch_reward + (1.0 - batch_done) * self.cfg.gamma * double_q
        else:
            Q_next = sess.run(self.main_net.Q, feed_dict={
                self.main_net.states: batch_next_state, self.main_net.dropout: 1.0
            })
            Q_target = batch_reward + (1.0 - batch_done) * self.cfg.gamma * np.max(Q_next, axis=1)

        _, loss, summary = sess.run([self.apply_gradients, self.main_net.loss, self.train_summary], feed_dict={
            self.main_net.states: batch_state,
            self.main_net.actions: batch_action,
            self.main_net.Q_target: Q_target,
            self.main_net.dropout: self.cfg.dropout
        })

        if self.global_t % 10 == 0:
            self.train_summary_writer.add_summary(summary, self.global_t)
            self.train_summary_writer.flush()

        if self.cfg.use_double_dqn and self.global_t % self.cfg.net_update_step == 0:
            sess.run(self.sync_target_net)
        return

    def train(self, saver, sess, env):
        cfg = self.cfg
        process_fn = create_process_fn(cfg.env_mode, cfg.use_rgb)

        # summary
        self.add_train_summary(sess)
        self.global_t, self.n_episode = restore_session(saver, sess, cfg.model_dir)
        self.epsilon = self._anneal_epsilon(self.global_t)

        epi_rewards = []
        best_epi_reward = 0.0
        while not self.stop_requested and self.global_t < cfg.max_train_step:
            self.n_episode += 1
            self.logger.info("\n-------new epoch: {}----------".format(self.n_episode))

            o_t = env.reset()
            o_t = process_fn(o_t)
            s_t = np.concatenate([o_t] * self.cfg.state_history, axis=2)
            done = False

            epi_reward = 0.0
            while not done and not self.stop_requested and self.global_t < cfg.max_train_step:
                action, action_q = self.pick_action(sess, s_t, reward=0.0, use_epsilon_greedy=True)
                o_t1, reward, done = env.step(action)
                o_t1 = process_fn(o_t1)
                s_t1 = np.concatenate([s_t[:, :, 3 if cfg.use_rgb else 1:], o_t1], axis=2)

                epi_reward += reward
                # if reward != 0 or done:
                #     self.logger.info("reward {} /done{}".format(reward, done))
                #     Image.fromarray(np.reshape(o_t1, [84, 84])) \
                #         .save("%s/%d.png" % (cfg.log_dir, self.global_t))
                #     Image.fromarray(img).save("%s/%d_o.png" % (cfg.log_dir, self.global_t))

                self._perceive(sess, s_t, action, reward, s_t1, done)
                if len(self.replay_buffer) > self.cfg.batch_size * 4:
                    self._update_weights(sess)
                if self.global_t % 100000 == 0:
                    backup_session(saver, sess, cfg.model_dir, self.global_t, self.n_episode)

                s_t = s_t1
                if self.global_t % 100 == 0 or reward > 0.0:
                    self.logger.info("global_t={} / action_id={} reward={:04.2f} / epsilon={:06.4f} / Q={:04.2f}"
                                     .format(self.global_t, action, reward, self.epsilon, action_q))

            # save the episode reward
            epi_rewards.append(epi_reward)
            if len(epi_rewards) % 100 == 0:
                self.logger.info("\n---episode={} / avg_epi_reward={:04.2f}".
                                 format(self.n_episode, np.mean(epi_rewards)))
                epi_rewards = []

            if epi_reward > best_epi_reward:
                self.logger.info("\n===reward improved from {:04.2f} to {:04.2f}".format(best_epi_reward, epi_reward))
                best_epi_reward = epi_reward
        backup_session(saver, sess, cfg.model_dir, self.global_t, self.n_episode)
        return
