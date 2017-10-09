from __future__ import print_function

from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
from util.feature_util import process_image, normalize
from PIL import Image


def main():
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    # print(p.getActionSet())
    # agent = myAgentHere(allowed_actions=p.getActionSet())

    p.init()
    p.act(p.getActionSet()[0])
    reward = 0.0
    nb_frames = 50
    for i in range(nb_frames):
        if p.game_over():
            p.reset_game()

        o_t = p.getScreenRGB()
        o_t = process_image(o_t, (84, 84), None, False)
        Image.fromarray(np.reshape(o_t, [84, 84])).save("tmp_%d.png" % (i))
        # Image.fromarray(o_t).save("tmp_%d.png" % (i))
        # action = agent.pickAction(reward, observation)
        action = random.choice(p.getActionSet())
        reward = p.act(action)
        reward = np.clip(reward, -1.0, 1.0)
        print(reward)


if __name__ == '__main__':
    main()
