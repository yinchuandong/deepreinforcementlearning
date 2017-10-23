import numpy as np


class Environment(object):

    def __init__(self, env_name, mode="atari", display=True, frame_skip=1):
        self.mode = mode
        self.display = display

        if self.mode == "atari":
            # atari game
            import gym
            from gym.wrappers.frame_skipping import SkipWrapper
            self.env = gym.make(env_name)
            skip_wrapper = SkipWrapper(frame_skip)
            self.env = skip_wrapper(self.env)
            self._action_size = self.env.action_space.n
            self._action_set = [k for k in range(self._action_size)]
        elif self.mode == "ple":
            # ple game
            from ple.games import *
            from ple import PLE
            game = eval(env_name)()
            self.env = PLE(game, fps=30, display_screen=display, frame_skip=frame_skip)
            self._action_size = len(self.env.getActionSet())
            self._action_set = self.env.getActionSet()
        elif self.mode == "custom":
            from customgame import *
            self.env = eval(env_name)(display_screen=display, fps=60, frame_skip=frame_skip)
            self._action_size = self.env.action_size
            self._action_set = self.env.action_set
        else:
            raise Exception("mode can only be [atari, ple, custom]")
        return

    @property
    def action_size(self):
        return self._action_size

    @property
    def action_set(self):
        return self._action_set

    def step(self, action_idx):
        if self.mode == "atari":
            img, reward, terminal, _ = self.env.step(action_idx)
            if self.display:
                self.env.render()
        elif self.mode == "ple":
            action = self.action_set[action_idx]
            reward = self.env.act(action)
            img = self.env.getScreenRGB()
            img = np.rot90(img, 3)
            terminal = self.env.game_over()
        elif self.mode == "custom":
            img, reward, terminal = self.env.step(action_idx)
            img = np.rot90(img, 3)
        return (img, reward, terminal)

    def reset(self):
        if self.mode == "atari":
            img = self.env.reset()
        elif self.mode == "ple":
            self.env.reset_game()
            img = self.env.getScreenRGB()
            img = np.rot90(img, 3)
        elif self.mode == "custom":
            img = self.env.reset()
            img = np.rot90(img, 3)
        return img
