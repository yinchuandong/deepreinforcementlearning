from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from feature_util import process_image, normalize, create_process_fn
from PIL import Image
import matplotlib.pyplot as plt
from time import time
from pympler.asizeof import asizeof
from collections import deque
from copy import deepcopy


def test_crop_vision():
    # env = gym.make('Breakout-v0')
    # o_t = env.reset()
    # print(o_t.shape)
    # o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), False)
    # o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), True)
    # print(o_t.dtype, asizeof(o_t))
    # o_t2 = normalize(o_t)
    # print(o_t2.dtype, asizeof(o_t2))
    # s_t = np.concatenate([o_t, o_t, o_t, o_t], axis=2)
    # print(s_t.dtype, asizeof(s_t))

    # s_t1 = s_t
    # print('s_t', id(s_t))
    # print('s_t1', id(s_t1))
    # e_t = (s_t, 0)
    # e_t1 = (s_t1, 1)
    # print('-----', s_t[0, 0, 0])
    # s_t[0, 0, 0] = 111
    # s_t, _ = e_t
    # s_t1, _ = e_t1
    # print('-----', s_t1[0, 0, 0])
    # print('s_t', id(s_t))
    # print('s_t1', id(s_t1))
    # print('e_t', id(e_t), asizeof(e_t))
    # print('e_t1', id(e_t1), asizeof(e_t1))

    # return
    # size = 1000
    # replay_buffer = deque(maxlen=size)
    # for k in range(size):
    #     replay_buffer.append(deepcopy(o_t))

    # print("buffer size:", asizeof(replay_buffer))

    return


def test_atari_img_process():
    process_fn = create_process_fn(is_atari=False, use_rgb=True)
    o_t = Image.open("tmp2.png")
    o_t = np.array(o_t)

    o_t = process_fn(o_t)
    # print(o_t[o_t == 0].shape)
    # print(o_t[o_t != 0].shape)
    # print(o_t.shape)
    plt.figure(figsize=(3, 3))
    plt.imshow(o_t)
    plt.show()

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
    test_atari_img_process()
    # test_crop_vision()
    # test_gym_render()
    return


if __name__ == '__main__':
    main()
