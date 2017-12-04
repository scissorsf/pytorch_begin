
from __future__ import division

import gc
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from torch.autograd import Variable

import buffer
import train

#env = gym.make('BipedalWalker-v2')
#env = gym.make('Pendulum-v0')
env = gym.make('InjectWorld-v1')

MAX_EPISODES = 2000
MAX_STEPS = 100
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print(' State Dimensions :- ', S_DIM)
print(' Action Dimensions :- ', A_DIM)
print(' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)
def main():
    plt.ion()
    ep_x = []
    re_y = []
    for _ep in range(MAX_EPISODES):
        observation = env.reset()
        
        result = 0
        for r in range(MAX_STEPS):
            #env.render()
            state = np.float32(observation)

            action = trainer.get_exploration_action(state)
            # if _ep%5 == 0:
            # 	# validate every 5th episode
            # 	action = trainer.get_exploitation_action(state)
            # else:
            # 	# get action based on observation, use exploration policy here
            # 	action = trainer.get_exploration_action(state)

            new_observation, reward, done, info = env.step(action)

            # # dont update if this is validation
            # if _ep%50 == 0 or _ep>450:
            # 	continue
            result += reward
            if done:
                new_state = None
            else:
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)

            observation = new_observation

            # perform optimization
            trainer.optimize()
            if done:
                
                break
        ep_x.append(_ep)
        re_y.append(result)
        print('EPISODE :- ', _ep,result)
        # check memory consumption and clear memory
        #gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)

        #if _ep % 100 == 0:
        #	trainer.save_models(_ep)
        plt.cla()
        plt.plot(ep_x,re_y)
        plt.pause(0.0001)
    plt.ioff()  
    plt.show()
    #plt.savefig("examples.jpg")
    print('Completed episodes')

if __name__ == '__main__':
    main()
