from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from util.feature_util import process_image, normalize
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from pympler.asizeof import asizeof
from collections import deque
from copy import deepcopy


def test_crop_vision():
    env = gym.make('Breakout-v0')
    o_t = env.reset()
    print(o_t.shape)
    # o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), False)
    o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), True)
    print(o_t.dtype, asizeof(o_t))
    o_t2 = normalize(o_t)
    print(o_t2.dtype, asizeof(o_t2))

    size = 1000
    replay_buffer = deque(maxlen=size)
    for k in range(size):
        replay_buffer.append(deepcopy(o_t))

    print("buffer size:", asizeof(replay_buffer))
    # o_t = process_image(o_t, (84, 110), None, True)
    # print(o_t.shape)
    # plt.figure(figsize=(3, 3))
    # plt.imshow(o_t)
    # plt.show()
    return


def test_gym_render():
    env = gym.make('Breakout-v0')
    env.reset()
    t_start = time()
    for epoch in range(200):
        # env.render()
        env.step(env.action_space.sample())
    t_end = time()
    print('time', (t_end - t_start))
    return


def main():
    test_crop_vision()
    # test_gym_render()
    return


if __name__ == '__main__':
    main()
