from __future__ import print_function

# from ple.games.flappybird import FlappyBird
from ple.games import *
from ple import PLE
import random
import numpy as np
from util.feature_util import normalize, create_process_fn
from PIL import Image
from util.generic_util import get_logger
from util.env_util import Environment


def test1():
    # game = FlappyBird()
    game = eval('flappybird.FlappyBird')()
    p = PLE(game, fps=30, display_screen=True, frame_skip=3)
    # print(p.getActionSet())
    # agent = myAgentHere(allowed_actions=p.getActionSet())
    logger = get_logger("tmp_dqn/tmp.log")

    p.init()
    p.act(p.getActionSet()[0])
    reward = 0.0
    nb_frames = 50
    for i in range(nb_frames):
        if p.game_over():
            p.reset_game()

        o_t = p.getScreenRGB()
        o_t = process_image(o_t, (84, 84), None, False)
        # Image.fromarray(np.reshape(o_t, [84, 84])).save("tmp_%d.png" % (i))
        # Image.fromarray(o_t).save("tmp_%d.png" % (i))
        # action = agent.pickAction(reward, observation)
        action = random.choice(p.getActionSet())
        reward = p.act(action)
        reward = np.clip(reward, -1.0, 1.0)
        # print(reward)
        logger.info("{}".format(i))
    return


def test2():
    logger = get_logger("tmp_dqn/tmp.log")
    process_fn = create_process_fn(is_atari=False, use_rgb=True)
    env = Environment('CustomFlappyBird', mode="custom", display=False)
    # env = Environment('FlappyBird', mode="ple", frame_skip=10, display=True)
    # env = Environment('Breakout-v0', mode="atari", frame_skip=10, display=True)
    nb_frames = 100
    env.reset()
    for i in range(nb_frames):
        # Image.fromarray(np.reshape(o_t, [84, 84])).save("tmp_%d.png" % (i))
        # action = agent.pickAction(reward, observation)
        action = np.random.randint(0, env.action_size)
        o_t, reward, terminal = env.step(action)
        reward = np.clip(reward, -1.0, 1.0)
        o_t = process_fn(o_t)
        print(o_t.shape, o_t.dtype)
        Image.fromarray(o_t).save("tmp_%d.png" % (i))
        # print(reward)
        logger.info("{}".format(i))
        if terminal:
            env.reset()

    return


def test3():
    from customgame import CustomFlappyBird

    # env = CustomFlappyBird(display_screen=False)
    env = CustomFlappyBird(display_screen=True)
    for _ in range(60):
        action = np.random.randint(0, env.action_size)
        o_t, reward, terminal = env.step(action)
        print(action, reward, terminal)
    return


def main():
    test2()
    # test3()
    return


if __name__ == '__main__':
    main()
