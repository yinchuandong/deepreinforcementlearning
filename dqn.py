from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading
import signal
import gym

from dqn.agent import Agent


class Application(object):

    def __init__(self, config):
        # env = gym.make("Enduro-v0")
        self.env = gym.make(config.env_name)
        config.model_dir = config.save_dir + "/models"
        config.log_dir = config.save_dir + "/logs"
        config.action_dim = self.env.action_space.n

        self.config = config

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.agent = Agent(self.config)
            self.saver = tf.train.Saver()
            sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
            self.sess = tf.Session(config=sess_config)
            self.sess.run(tf.global_variables_initializer())
        return

    def signal_handler(self, signal_, frame_):
        print("You pressed Ctrl+C !")
        self.agent.stop_requested = True
        return

    def train(self):
        with self.graph.as_default():
            self.agent.train(self.saver, self.sess, self.env)
        return

    def run(self):
        train_thread = threading.Thread(target=self.train)
        signal.signal(signal.SIGINT, self.signal_handler)
        train_thread.start()

        print("Press Ctrl+C to stop")
        signal.pause()
        print("Now saving data....")
        train_thread.join()


def main(args):
    config = tf.app.flags.FLAGS
    app = Application(config)
    app.train()
    # app.run()
    return


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("env_name", "Breakout-v0", "the name of game to be trained")
    tf.app.flags.DEFINE_string("save_dir", "tmp_dqn", "save models and logs")
    tf.app.flags.DEFINE_boolean("use_gpu", False, "use gpu or cpu to train")
    tf.app.flags.DEFINE_integer("max_train_step", 10 * 10 ** 7, "max steps to train")
    tf.app.flags.DEFINE_integer("replay_size", 1 * 10 ** 5, "the size of replay buffer")

    tf.app.flags.DEFINE_boolean("use_double_dqn", False, "whether use target net to estimate Q_target")
    tf.app.flags.DEFINE_integer("net_update_step", 1000, "the update step of target net")
    tf.app.flags.DEFINE_boolean("use_duel_dqn", False, "whether use duelling channel")
    tf.app.flags.DEFINE_boolean("use_rgb", False, "whether use rgb or gray image")
    tf.app.flags.DEFINE_integer("frame_skip", 4, "the number of skipping frames")
    tf.app.flags.DEFINE_integer("state_dim", 84, "the width and height of state")
    tf.app.flags.DEFINE_integer("state_chn", 4, "the channel of state")
    # tf.app.flags.DEFINE_integer("action_dim", 5, "the action size of game")

    tf.app.flags.DEFINE_integer("epsilon_timestep", 1 * 10 ** 5, "the step of epsilon greedy")
    tf.app.flags.DEFINE_float("epsilon_hi", 1.0, "maximum epsilon greedy")
    tf.app.flags.DEFINE_float("epsilon_lo", 0.1, "minimum epsilon greedy")
    tf.app.flags.DEFINE_integer("batch_size", 32, "batch_size")

    tf.app.flags.DEFINE_float("gamma", 0.99, "the discounted factor of reward")
    tf.app.flags.DEFINE_float("lr", 1e-3, "learning rate")
    tf.app.flags.DEFINE_float("lr_decay", 0.99, "learning rate decay")
    tf.app.flags.DEFINE_float("max_gradient", 1.0, "maximum gradient when clipping gradients")
    tf.app.flags.DEFINE_float("dropout", 0.5, "the keep prob of dropout")
    tf.app.run()
