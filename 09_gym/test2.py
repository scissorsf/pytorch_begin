# -*- coding: utf-8 -*-
"""
filename: test2.py
Created on: 2017-11-26
@author: Gfei
"""

import gym
import numpy as np
env = gym.make('InjectWorld-v1')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

state = env.reset()
print(state)
print(env.action_space.sample())
for i in range(10):
    s,r,d,_ = env.step(env.action_space.sample())
    print(s,r,d)
