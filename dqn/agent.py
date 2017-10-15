from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from collections import deque

from base.base_agent import BaseAgent
from util.network_util import restore_session, backup_session
from util.feature_util import process_image, normalize
from .network import Network


class Agent(BaseAgent):

    def __init__(self, cfg):
        BaseAgent.__init__(self)

        self.cfg = cfg

        # create networks
        state_chn = cfg.state_chn * (3 if cfg.use_rgb else 1)
        input_shape = [cfg.state_dim, cfg.state_dim, state_chn]
        device = "/gpu:0" if cfg.use_gpu else "/cpu:0"
        self.main_net = Network(input_shape, cfg.action_dim, "main_net", device)
        self.target_net = Network(input_shape, cfg.action_dim, "target_net", device)
        self.sync_target_net = self.target_net.sync_from(self.main_net)

        # create gradients operations
        optimizer = tf.train.RMSPropOptimizer(cfg.lr, decay=cfg.lr_decay)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, cfg.max_grad) for grad in gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        self.replay_buffer = deque(maxlen=cfg.replay_size)
        self.stop_requested = False

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

    def train_ple(self, saver, sess, env):
        cfg = self.cfg
        # summary
        self.add_train_summary(sess)
        self.global_t = restore_session(saver, sess, cfg.model_dir)
        self.epsilon = self._anneal_epsilon(self.global_t)

        env.init()
        while not self.stop_requested and self.global_t < cfg.max_train_step:
            print("-------new epoch-----------------")
            env.reset_game()
            env.act(env.getActionSet()[0])
            o_t = env.getScreenRGB()
            o_t = process_image(o_t, (84, 84), None, cfg.use_rgb)
            o_t = normalize(o_t)
            s_t = np.concatenate([o_t, o_t, o_t, o_t], axis=2)
            done = False
            # last_action = None

            local_t = 0
            while not done and not self.stop_requested and self.global_t < cfg.max_train_step:
                local_t += 1
                action, action_q = self.pick_action(sess, s_t, reward=0.0, use_epsilon_greedy=True)
                reward = env.act(env.getActionSet()[action])
                reward = np.clip(reward, -1.0, 1.0)
                o_t1 = env.getScreenRGB()
                done = env.game_over()

                o_t1 = process_image(o_t1, (84, 84), None, cfg.use_rgb)
                o_t1 = normalize(o_t1)
                # Image.fromarray(np.reshape(o_t1, [84, 84])).save("tmp/%d.png" % (self.global_t))
                s_t1 = np.concatenate([s_t[:, :, 3 if cfg.use_rgb else 1:], o_t1], axis=2)

                self._perceive(sess, s_t, action, reward, s_t1, done)
                if len(self.replay_buffer) > self.cfg.batch_size * 4:
                    self._update_weights(sess)
                if self.global_t % 100000 == 0:
                    backup_session(saver, sess, cfg.model_dir, self.global_t)

                s_t = s_t1
                # last_action = action
                if self.global_t % 100 == 0 or reward > 0.0:
                    print("global_t=%d / action_id=%d reward=%.2f / epsilon=%.6f / Q=%.4f"
                          % (self.global_t, action, reward, self.epsilon, action_q))

        backup_session(saver, sess, cfg.model_dir, self.global_t)
        return

    def train_atari(self, saver, sess, env):
        cfg = self.cfg
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
            # last_action = None

            local_t = 0
            while not done and not self.stop_requested and self.global_t < cfg.max_train_step:
                env.render()
                local_t += 1
                # frame skipping
                # if local_t % cfg.frame_skip != 0 and last_action is not None:
                #     action_q = -1  # skipping
                #     o_t1, reward, done, info = env.step(last_action)
                # else:
                #     action, action_q = self.pick_action(sess, s_t, reward=0.0, use_epsilon_greedy=True)
                #     o_t1, reward, done, info = env.step(action)
                action, action_q = self.pick_action(sess, s_t, reward=0.0, use_epsilon_greedy=True)
                o_t1, reward, done, info = env.step(action)
                o_t1 = process_image(o_t1, (110, 84), (0, 20, cfg.state_dim, 20 + cfg.state_dim), cfg.use_rgb)
                # Image.fromarray(np.reshape(o_t1, [84, 84])).save("tmp/%d.png" % (self.global_t))
                s_t1 = np.concatenate([s_t[:, :, 3 if cfg.use_rgb else 1:], o_t1], axis=2)

                # reward reshaping
                if reward == 0.0:
                    reward == 0.1
                if done:
                    reward = -1.0

                self._perceive(sess, s_t, action, reward, s_t1, done)
                if len(self.replay_buffer) > self.cfg.batch_size * 4:
                    self._update_weights(sess)
                if self.global_t % 100000 == 0:
                    backup_session(saver, sess, cfg.model_dir, self.global_t)

                s_t = s_t1
                # last_action = action
                if self.global_t % 100 == 0 or reward > 0.0:
                    print("global_t=%d / action_id=%d reward=%.2f / epsilon=%.6f / Q=%.4f"
                          % (self.global_t, action, reward, self.epsilon, action_q))

        backup_session(saver, sess, cfg.model_dir, self.global_t)
        return
