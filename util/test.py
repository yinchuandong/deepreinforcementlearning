from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from imgutil import process_image
from PIL import Image
import matplotlib.pyplot as plt


def test_crop_vision():
    env = gym.make('Breakout-v0')
    o_t = env.reset()
    print(o_t.shape)
    # o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), False)
    o_t = process_image(o_t, (84, 110), (0, 20, 84, 104), True)
    # o_t = process_image(o_t, (84, 110), None, True)
    print(o_t.shape)
    plt.figure(figsize=(3, 3))
    plt.imshow(o_t)
    plt.show()
    return


def main():
    test_crop_vision()
    return


if __name__ == '__main__':
    main()
