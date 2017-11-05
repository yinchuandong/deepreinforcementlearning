from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import threading
import signal
from pg.agent import Agent
from util.generic_util import get_logger
from util.env_util import Environment


class Application(object):

    def __init__(self, cfg):
        cfg.model_dir = cfg.save_dir + "/models"
        cfg.log_dir = cfg.save_dir + "/logs"
        cfg.log_filename = cfg.log_dir + "/debug.log"

        tf.set_random_seed(cfg.random_seed)
        random.seed(cfg.random_seed)

        self.logger = get_logger(cfg.log_filename)

        # atari game
        # self.env = Environment(cfg.env_name, cfg.env_mode, cfg.display, cfg.frame_skip)
        # cfg.action_dim = self.env.action_size
        import gym
        self.env = gym.make('CartPole-v0').unwrapped
        cfg.action_dim = 2
        cfg.state_dim = 4

        self.cfg = cfg
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.agent = Agent(self.cfg, self.logger)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            sess_config = tf.ConfigProto(
                log_device_placement=False,
                allow_soft_placement=True,
                gpu_options=gpu_options
            )
            self.sess = tf.Session(config=sess_config)
            self.sess.run(tf.global_variables_initializer())
        return

    def signal_handler(self, signal_, frame_):
        self.logger.info("You pressed Ctrl+C !")
        self.agent.stop_requested = True
        return

    def train(self):
        with self.graph.as_default():
            self.agent.train(self.sess, self.env)
        return

    def run(self):
        train_thread = threading.Thread(target=self.train)
        signal.signal(signal.SIGINT, self.signal_handler)
        train_thread.start()

        self.logger.info("Press Ctrl+C to stop")
        signal.pause()
        self.logger.info("Now saving data....")
        train_thread.join()


def main(args):
    cfg = tf.app.flags.FLAGS
    app = Application(cfg)
    # app.train()
    app.run()
    return


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("env_name", "CustomFlappyBird", "the name of game to be trained")
    tf.app.flags.DEFINE_string("env_mode", "custom", "in [atari, ple, custom]")
    tf.app.flags.DEFINE_string("save_dir", "tmp_pg", "save models and logs")
    tf.app.flags.DEFINE_boolean("use_gpu", True, "use gpu or cpu to train")
    tf.app.flags.DEFINE_integer("max_train_step", 10 * 10 ** 7, "max steps to train")

    tf.app.flags.DEFINE_integer('random_seed', 666, 'Value of random seed')

    tf.app.flags.DEFINE_boolean("train", True, "train or test")
    tf.app.flags.DEFINE_boolean("display", True, "whether display the enviroment")
    tf.app.flags.DEFINE_integer("frame_skip", 1, "the number of skipping frames")

    tf.app.flags.DEFINE_boolean("use_rgb", False, "whether use rgb or gray image")
    tf.app.flags.DEFINE_integer("state_dim", 84, "the width and height of state")
    tf.app.flags.DEFINE_integer("state_history", 4, "the number of consecutive frames as feature")

    tf.app.flags.DEFINE_integer("batch_size", 200, "batch_size")

    tf.app.flags.DEFINE_float("gamma", 0.99, "the discounted factor of reward")
    tf.app.flags.DEFINE_float("lr", 0.02, "learning rate")
    tf.app.flags.DEFINE_float("lr_decay", 0.99, "learning rate decay")
    tf.app.flags.DEFINE_float("max_grad", 10.0, "maximum gradient when clipping gradients")
    tf.app.flags.DEFINE_float("dropout", 1.0, "the keep prob of dropout")
    tf.app.run()
