# -*- coding: utf-8 -*-
"""
filename: injectworld_2.py
Created on: 2017-12-26
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


class InjectWorldEnv_2(gym.Env):

    def __init__(self):

        #parameters's space
        self.current_para = np.zeros(3)
        self.parameter_min = 50.
        self.parameter_max = 200.
        self._goal = 105
        self._optimal_range = 10
        self._optimal_min = -10 + self._goal
        self._optimal_max = 10 + self._goal

        #parameters's adjustment
        self.periods = 50.
        self.adjustment_max = 1.
        self.adjustment_min = -1.

        #each dimension's value of state is in list [-1, -0.5, 0, 0.5 , 1]
        self.state = []
        self.state_min = 0.
        self.state_med = 0.5
        self.state_max = 1.

        self.steps_before_done = 0
        self.action_space = spaces.Box(
            self.adjustment_min, self.adjustment_max, (3,))
        self.observation_space = spaces.Box(-self.state_max,
                                            self.state_max, (3,))

        self.reset()

    def _reset(self):

        self.state = [0, 0, 0]
        #judge = []
        #while True:
        #    self.state = np.random.choice([-1, -0.5, 0, 0.5, 1],size=(3,)).tolist()
        #    done = self.get_judge(judge)
        #    if not done:
        #        return np.array(self.state)
        while True:

            self.steps_before_done = 0
            self.current_para = self.init_parameters()  # [50,50,50]
            print(self.current_para)
            for i in range(3):
                self.state[i] = self.is_defect(self.current_para[i])
            done = self.get_judge()

            if not done:
                return np.array(self.state)

    def init_parameters(self):
        pa_range = np.arange(self.parameter_min, self.parameter_max + 1)
        return np.random.choice(pa_range, size=(3,))

    def get_judge(self):
        judgelist = []
        for i in range(3):
            if self.state[i] == 0:
                judgelist.append(True)
            else:
                judgelist.append(False)
        done = bool(all(judgelist))
        return done

    def is_defect(self, c_p):

        delta = c_p - self._goal

        if delta <= 0:
            negative = True
        else:
            negative = False

        if abs(delta) <= self._optimal_range:
            return 0.

        if not negative:
            if (delta - 10) / (self.parameter_max - self._optimal_max) <= 0.5:
                return 0.5
            else:
                return 1.
        else:
            if(-delta - 10) / (self._optimal_min - self.parameter_min) <= 0.5:
                return -0.5
            else:
                return -1.

    def get_c_s(self):

        return self.state

    def get_c_p(self):

        return self.current_para

    def get_goal(self):

        return [self._optimal_min, self._optimal_max]

    def _step(self, action):

        self.steps_before_done += 1
        self.current_para += np.array(action) * self.periods

        #clip parameters to ensure in right range
        for i in range(3):
            if self.current_para[i] > self.parameter_max:
                self.current_para[i] = self.parameter_max
            elif self.current_para[i] < self.parameter_min:
                self.current_para[i] = self.parameter_min

        #calculate defects
        for i in range(3):

            self.state[i] = self.is_defect(self.current_para[i])

        done = self.get_judge()

        reward = 0
        state = self.state
        if done:
            reward = 100.
            return np.array(self.state), reward, done, {}

        #reward -= np.sum(np.array(action)**2)*0.1
        #reward -= 1.

        reward -= np.sum(np.abs(np.array(state))) / 2 + \
            self.steps_before_done * 0.01

        return np.array(self.state), reward, done, {}
