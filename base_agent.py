import tensorflow as tf


class BaseAgent(object):

    def __init__(self):
        return

    def step(self, action):
        raise NotImplementedError('please override step function in BaseAgent')

    def create_summary(self, summary_dir, sess):
        self.reward_input = tf.placeholder(tf.float32)
        sum_reward = tf.summary.scalar('reward', self.reward_input)

        self.loss_input = tf.placeholder(tf.float32)
        sum_total = tf.summary.scalar('loss', self.loss_input)

        self.summary_op = tf.summary.merge([sum_reward, sum_total])
        self.summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        return