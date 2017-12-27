# -*- coding: utf-8 -*-
"""
filename: test2.py
Created on: 2017-11-26
@author: Gfei
"""

import gym
import numpy as np
import os 
import time
env = gym.make('InjectWorld-v1')
env = env.unwrapped
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high
print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

state = env.reset()

print(state)
print(env.action_space.sample())
print(env.get_goal())
total_r = 0
for i in range(10):

    s,r,d,_ = env.step([2,2,2])
    #os.system('pause')
    #time.sleep(3)
    total_r += r
    print('step:',i,s,r,d)
    if d:
        break

print(total_r)
