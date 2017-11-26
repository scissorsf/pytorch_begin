# -*- coding: utf-8 -*-
"""
filename: myGridWorld.py
Created on: 2017-1-1
@author: Gfei
"""
''' 
    第一步：将我们自己的环境文件（我创建的文件名为grid_mdp.py)拷贝到你的gym安装目录/gym/gym/envs/classic_control文件夹中。（拷贝在这个文件夹中因为要使用rendering模块。当然，也有其他办法。该方法不唯一）

    第二步：打开该文件夹（第一步中的文件夹）下的__init__.py文件，在文件末尾加入语句：
    from gym.envs.classic_control.myGridWorld import GridEnv

    第三步：进入文件夹你的gym安装目录/gym/gym/envs，打开该文件夹下的__init__.py文件，添加代码：
    register(
        id='GridWorld-v0',
        entry_point='gym.envs.classic_control:GridEnv',
        max_episode_steps=200,
        reward_threshold=100.0,
    ) 
'''
import logging
import random

import gym
import numpy
from gym import spaces

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):
    metadata= {
        'render.modes': ['human', 'rgb_array'],
        'video.frams_per_second':2
    }

    def __init__(self):
        
        self.states = [1,2,3,4,5,6,7,8]
        self.x = [140,220,300,380,460,140,300,460]
        self.y = [250, 250, 250, 250, 250, 150, 150,150 ]
        self.terminate_states = dict()
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        self.actions = ['n', 'e', 's','w']

        self.rewards = dict()
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        self.t = dict()
        self.t['1_s'] =6
        self.t['1_e'] =2
        self.t['2_w'] =1
        self.t['2_e'] =3
        self.t['3_w'] =2
        self.t['3_e'] =4
        self.t['3_w'] =7
        self.t['4_w'] =3
        self.t['4_e'] =5
        self.t['5_w'] =4
        self.t['5_s'] =8

        self.gamma = 0.8
        self.viewer =None
        self.state = None

    def getTerminal(self):
        return self.terminate_states

    def getGamma(self):
        return self.gamma
    
    def getStates(self):
        return self.states

    def getAction(self):
        return self.actions

    def getTerminal_states(self):
        return self.terminate_states

    def setAction(self,s):
        self.state = s

    def _step(self, action):
        state = self.state
        if state in self.terminate_states:
            return state, 0, True, {}

        key = "{:d}_{:s}".format(state, action)

        if key in self.t:
            next_state =self.t[key]
        else:
            next_state = state
        self.state = next_state

        is_terminal = False

        if next_state in self.terminate_states:
            is_terminal = True
        
        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]

        return next_state, r, is_terminal, {}

    def _reset(self):
        self.state = self.states[random.randint(0,4)]
        return self.state

    def render(self,mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.line1 = rendering.Line((100, 300), (500, 300))
            self.line2 = rendering.Line((100, 200), (500, 200))
           
            self.line3 = rendering.Line((500, 300), (500, 100))
            self.line4 = rendering.Line((100, 300), (100, 100))
            self.line5 = rendering.Line((180, 300), (180, 100))
            self.line6 = rendering.Line((260, 300), (260, 100))
            self.line7 = rendering.Line((340, 300), (340, 100))
            self.line8 = rendering.Line((420, 300), (420, 100))

            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))
            lines = [self.line1, self.line2,
                     self.line3, self.line4, self.line5, self.line6, self.line7, self.line8, self.line9, self.line10, self.line11 ]
            self.skull_1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140,150))
            self.skull_1.add_attr(self.circletrans)
            self.skull_1.set_color(0,0,0)

            self.skull_2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.skull_2.add_attr(self.circletrans)
            self.skull_2.set_color(0, 0, 0)

            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)

            self.robot = rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)
            
            for i in range(11):
                lines[i].set_color(0,0,0)
                self.viewer.add_geom(lines[i])
            self.viewer.add_geom(self.skull_1)
            self.viewer.add_geom(self.skull_2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        #self.robotrans.set_translation(self.x[self.state-1],self.y[self.state-1])
        self.robotrans.set_translation(self.x[self.state - 1], self.y[self.state - 1])

        return self.viewer.render(return_rgb_array=mode =='rgb_array')
