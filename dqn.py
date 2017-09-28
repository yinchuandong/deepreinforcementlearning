from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from dqn.trainer import Trainer


def main(args):
    config = tf.app.flags.FLAGS
    trainer = Trainer(config)
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
