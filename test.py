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
    use_rgb = False
    env_mode = "custom"
    process_fn = create_process_fn(env_mode=env_mode, use_rgb=use_rgb)
    env = Environment('CustomFlappyBird', env_mode=env_mode, display=False, frame_skip=3)
    # env = Environment('FlappyBird', env_mode="ple", frame_skip=10, display=True)
    # env = Environment('Breakout-v0', env_mode="atari", frame_skip=10, display=True)
    nb_frames = 20
    env.reset()
    for i in range(nb_frames):
        # Image.fromarray(np.reshape(o_t, [84, 84])).save("tmp_%d.png" % (i))
        # action = agent.pickAction(reward, observation)
        action = np.random.randint(0, env.action_size)
        o_t, reward, terminal = env.step(action)
        reward = np.clip(reward, -1.0, 1.0)
        o_t = process_fn(o_t)
        # print(o_t.shape, o_t.dtype)
        if not use_rgb:
            o_t = np.reshape(o_t, [84, 84]) * 255
        # print(o_t)
        # print(o_t.dtype)
        Image.fromarray(o_t).save("tmp_%d.png" % (i))
        # print(reward)
        logger.info("{}/{}/{}".format(i, reward, terminal))
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


def test4():
    w1 = 8 * 8 * 16 * 32
    w2 = 4 * 4 * 32 * 64
    w3 = 3 * 3 * 64 * 64
    fc1 = 1000 * 512
    fc2 = 512 * 2

    all = w1 + w2 + w3 + fc1 + fc2
    print(all * 4 / 1024)
    return


def _discount_reward(rewards, gamma=0.99):
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        # if rewards[t] < 0:  # for specific game
        #     running_add = 0
        running_add = rewards[t] + running_add * gamma
        discounted_r[t] = running_add

    discounted_r -= discounted_r.mean()
    discounted_r /= discounted_r.std()
    return discounted_r


def test5():

    rewards = [0, 0, 0, 1]
    d_rewards = _discount_reward(rewards)
    print(d_rewards)
    return


def test6():
    import gym
    env = gym.make('CartPole-v0')
    env.seed(1)     # reproducible, general Policy gradient has high variance
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    o_t = env.reset()
    for _ in range(1):
        env.render()
        o_t1, reward, done, info = env.step(1)
        print(o_t1)
    return


def main():
    test2()
    # test3()
    # test4()
    # test5()
    # test6()
    return


if __name__ == '__main__':
    main()
