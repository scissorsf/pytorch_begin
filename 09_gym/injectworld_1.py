# -*- coding: utf-8 -*-
"""
filename: injectworld_1.py
Created on: 2017-11-27
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


class InjectWorldEnv(gym.Env):

    def __init__(self):
        
        self.state = []
        self.optimal_min = 0.
        self.optimal_max = 10.
        self.min_parameters = -20
        self.max_parameters = 20
        self.steps_before_done = 0
        self.reward = 0.
        self.max_steps = 10
        self.min_state = -100
        self.max_state = 100
        self.action_space = spaces.Box(
            self.min_parameters, self.max_parameters, (3, ))
        self.observation_space = spaces.Box(
            self.min_state, self.max_state, (3, ))

        self.reset()

    def _reset(self):
        self.reward = 0.
        self.steps_before_done = 0.
        self.state = np.random.uniform(self.min_state,self.max_state,(3,)).tolist()
        return np.array(self.state)

    
    #action is a 1*3  list each between 1~200
    #action[0] refer to parameter_1
    #action[1] refer to parameter_2
    #action[2] refer to parameter_3

    

    def _step(self, action):
        self.steps_before_done += 1
        if self.steps_before_done == 1:
            self.reward = 0.
        #get new state
        for i in range(3):
            self.state[i] += action[i]
            #all in range
            if self.state[i] > self.max_state:
                self.state[i] = self.max_state
            elif self.state[i] < self.min_state:
                self.state[i]=self.min_state
           
        #judge state
        judge = []
        for i in range(3):
            if self.optimal_min < self.state[i] < self.optimal_max:
                judge.append(True)
            else :
                judge.append(False)
        done = bool(all(judge))

        if self.steps_before_done == self.max_steps:
            self.reward = -1.
            return np.array(self.state), self.reward, True, {}

        if done:
            self.reward = 0.
        else:
            self.reward = -1.

        return np.array(self.state), self.reward, done, {}
