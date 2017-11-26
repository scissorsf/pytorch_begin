# -*- coding: utf-8 -*-
"""
filename: test1.py
Created on: 2017-11-23
@author: Gfei
"""
import time
import gym
env = gym.make('MyGridWorld-v0')

state = env.reset()
print(state)
env.render()
