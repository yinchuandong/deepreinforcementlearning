import gym
from gym.wrappers.frame_skipping import SkipWrapper
from ple.games import *
from ple import PLE


class Environment(object):

    def __init__(self, env_name, is_atari=True, display=True, frame_skip=1):
        self.is_atari = is_atari
        self.display = display

        if self.is_atari:
            # atari game
            self.env = gym.make(env_name)
            skip_wrapper = SkipWrapper(frame_skip)
            self.env = skip_wrapper(self.env)
            self._action_size = self.env.action_space.n
            self._action_set = [k for k in range(self._action_size)]
        else:
            # ple game
            game = eval(env_name)()
            self.env = PLE(game, fps=30, display_screen=display, frame_skip=frame_skip)
            self._action_size = len(self.env.getActionSet())
            self._action_set = self.env.getActionSet()
        return

    @property
    def action_size(self):
        return self._action_size

    @property
    def action_set(self):
        return self._action_set

    def step(self, action):
        if self.is_atari:
            img, reward, terminal, _ = self.env.step(action)
            if self.display:
                self.env.render()
        else:
            reward = self.env.act(action)
            img = self.env.getScreenRGB()
            terminal = self.env.game_over()
        return (img, reward, terminal)

    def reset(self):
        if self.is_atari:
            img = self.env.reset()
        else:
            self.env.reset_game()
            img = self.env.getScreenRGB()
        return img
