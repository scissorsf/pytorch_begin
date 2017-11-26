# -*- coding: utf-8 -*-
"""
filename: test2.py
Created on: 2017-11-26
@author: Gfei
"""

import gym
import numpy as np
env = gym.make('InjectWorld-v0')

state = env.reset()
for i in range(10):
    s,r, d, _ =env.step([0.,5.,11.])
    print(s,r,d)
