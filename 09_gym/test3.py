# -*- coding: utf-8 -*-
"""
filename: test3.py
Created on: 2017-12-26
@author: Gfei
"""
import gym
import numpy as np
import os
import time
env = gym.make('InjectWorld-v2')
env = env.unwrapped
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.low
print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Low :- ', A_MAX)

state = env.reset()
print(env.get_goal())
print(env.get_c_p())

print(state)
a0 = env.action_space.sample()
print(a0)

s,r,d,_ = env.step([1,1,1])
print(env.get_c_p())
print(s, r, d)
