# -*- coding: utf-8 -*-
"""
filename: injectionworld.py
Created on: 2017-11-23
@author: Gfei
"""

#simulator of injection process
#just a example
#In 3 parameters spaces, to get a high-quality product
import logging
import random

import gym
import numpy as np
from gym import spaces

logger = logging.getLogger(__name__)


class InjectWorld(gym.Env):

    def __init__(self):

        self.unterminal_states = []
        self.states_build()
        self.terminal_state = [0., 0., 0.]
        self.state = []
        self.optimal_min = 0.
        self.optimal_max = 10.
        self.min_parameters = -100
        self.max_parameters = 100
        self.steps_before_done = 0
        self.reward = 0.
        self.max_steps = 10
        self.action_space = spaces.Box(
            self.min_parameters, self.max_parameters, (1, 3))

        self.reset()

    def _reset(self):
        self.reward = 0.
        self.steps_before_done = 0
        self.state = random.choice(self.unterminal_states)
        return np.array(self.state)

    def states_build(self):
        for i in [-1., 0., 1.]:
            for j in [-1., 0., 1.]:
                for k in [-1., 0., 1.]:
                    self.unterminal_states.append([i, j, k])
        self.unterminal_states.remove([0, 0, 0])
    #action is a 1*3  list each between 1~200
    #action[0] refer to parameter_1
    #action[1] refer to parameter_2
    #action[2] refer to parameter_3

    def _step(self, action):
        self.steps_before_done += 1
        if self.steps_before_done == 1:
            self.reward = 0

        #set state
        for i in range(3):
            if action[i] < self.optimal_min:
                self.state[i] = -1.
            elif action[i] > self.optimal_max:
                self.state[i] = 1.
            else:
                self.state[i] = 0.

        done = bool(self.state == self.terminal_state)

        if self.steps_before_done == self.max_steps:
            self.reward -= 1. * self.steps_before_done
            return np.array(self.state), self.reward, True, {}

        if done:
            self.reward = (self.max_steps - self.steps_before_done) * 10.
        else:
            self.reward -= 1. * self.steps_before_done

        return np.array(self.state), self.reward, done, {}
