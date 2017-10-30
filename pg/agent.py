from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
from PIL import Image

from base.base_agent import BaseAgent
from util.network_util import restore_session, backup_session
from util.feature_util import create_process_fn
from util.data_util import minibatches
from util.generic_util import Progbar
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
        self.main_net = Network(input_shape, cfg.action_dim, "main_net", device)

        self.logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for var in self.main_net.vars:
            print(var)
        self.logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        # create gradients operations
        optimizer = tf.train.RMSPropOptimizer(cfg.lr, decay=cfg.lr_decay)
        optimizer = tf.train.AdamOptimizer(cfg.lr)
        gradients = tf.gradients(self.main_net.loss, self.main_net.vars)
        gradients_clipped = [tf.clip_by_norm(grad, cfg.max_grad) for grad in gradients]
        self.apply_gradients = optimizer.apply_gradients(zip(gradients_clipped, self.main_net.vars))

        self.stop_requested = False
        self.global_t = 0
        self.n_episode = 0
        return

    def add_train_summary(self, sess):
        # tf.summary.scalar("loss", self.main_net.loss)
        self.train_summary = tf.summary.merge_all()
        self.train_summary_writer = tf.summary.FileWriter(self.cfg.log_dir + "/train", sess.graph)
        return

    def pick_action(self, sess, state):
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.cfg.action_dim)
            # action_idx = 0
        else:
            pi_out = sess.run(self.main_net.pi, feed_dict={
                self.main_net.states: [state], self.main_net.dropout: 1.0
            })[0]
            action_idx = np.random.choice(range(len(pi_out)), p=pi_out)
        self.epsilon = self._anneal_epsilon(self.global_t)
        return action_idx

    def _anneal_epsilon(self, timestep):
        cfg = self.cfg
        span = float(cfg.eps_hi - cfg.eps_lo) / float(cfg.eps_step)
        epsilon = cfg.eps_hi - span * min(timestep, cfg.eps_step)
        return epsilon

    def _discount_reward(self, rewards, gamma=0.99):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            # if rewards[t] < 0:  # for specific game
            #     running_add = 0
            running_add = rewards[t] + running_add * gamma
            discounted_r[t] = running_add

        # discounted_r -= discounted_r.mean()
        # discounted_r /= discounted_r.std()
        return discounted_r

    def _update_weights(self, sess, minibatch):
        batch_state, batch_action, batch_reward = [], [], []
        for s, a, r in minibatch:
            batch_state.append(s)
            batch_action.append(a)
            batch_reward.append(r)
        _, loss = sess.run([self.apply_gradients, self.main_net.loss], feed_dict={
            self.main_net.states: batch_state,
            self.main_net.actions: batch_action,
            self.main_net.rewards: batch_reward,
            self.main_net.dropout: self.cfg.dropout,
        })
        return loss

    def _run_episode(self, sess, epi_buffer):
        n_batches = (len(epi_buffer) + self.cfg.batch_size - 1) // self.cfg.batch_size
        prog = Progbar(target=n_batches)
        states, actions, rewards = [], [], []
        for s, a, r in epi_buffer:
            states.append(s)
            actions.append(a)
            rewards.append(r)
        discounted_rewards = self._discount_reward(rewards, self.cfg.gamma)
        epi_buffer = zip(states, actions, discounted_rewards)

        # print(discounted_rewards)
        # print(np.amax(states), np.amin(states))
        # print(actions)
        # print(len(actions))
        # import sys
        # sys.exit()
        for i, minibatch in enumerate(minibatches(epi_buffer, self.cfg.batch_size, False)):
            train_loss = self._update_weights(sess, minibatch)
            prog.update(i + 1, [("train_loss", train_loss)])
        # sys.exit()
        return

    def train(self, sess, env):
        cfg = self.cfg
        saver = tf.train.Saver(tf.global_variables())
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

            epi_buffer = []
            epi_reward = 0.0
            while not done and not self.stop_requested and self.global_t < cfg.max_train_step:
                self.global_t += 1
                action = self.pick_action(sess, s_t)
                o_t1, reward, done = env.step(action)
                o_t1 = process_fn(o_t1)
                s_t1 = np.concatenate([s_t[:, :, 3 if cfg.use_rgb else 1:], o_t1], axis=2)

                epi_buffer.append((s_t, action, reward))
                epi_reward += reward

                s_t = s_t1
                if self.global_t % 100 == 0 or reward > 0.0:
                    self.logger.info("global_t={} / action_idx={} reward={:04.2f} / epsilon={:06.4f} / pi={}"
                                     .format(self.global_t, action, reward, self.epsilon, str(action)))

            if self.stop_requested:
                # skip the uncomplete episode
                break
            # train per episode
            self._run_episode(sess, epi_buffer)
            epi_buffer = []

            if self.global_t % 100000 == 0:
                backup_session(saver, sess, cfg.model_dir, self.global_t, self.n_episode)

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
