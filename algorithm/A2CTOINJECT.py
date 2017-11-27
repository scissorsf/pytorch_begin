# -*- coding: utf-8 -*-
"""
filename: A2C.py
Created on: 2017-11-21
@author: Gfei
"""


import math
import os
import random
import utils
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Hyper parameter


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size,action_size,action_lim):
        super(ActorNetwork,self).__init__()

        self.action_lim = action_lim
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out= F.tanh(self.fc3(out)) 
        
        return out

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)
    
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
def roll_out(actor_network, task, sample_nums, value_network, init_state, noise,action_lim):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for j in range(sample_nums):
        states.append(state)
        #to be modified
        tanh_action = actor_network(Variable(torch.Tensor([state]))).detach()
        action = np.multiply(tanh_action.cpu().data.numpy()[
                              0], [20,20,20])
        #add  noise to explore
        ex_action = action + (noise.sample() * action_lim)
        #ex_action = action
        

        next_state, reward, done, _ = task.step(ex_action)
        actions.append(ex_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done :
            is_done = True
            state = task.reset()
            break
    if not is_done:
        state = task.reset()
        final_r = value_network(Variable(torch.Tensor([final_state]))).cpu().data.numpy()
        
    return states, actions, rewards, final_r, state

def discount_reward(r, gamma,final_r):
    discount_r = np.zeros_like(r)
    running_add= final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r


def plot_durations(plt, episode_durations, ave_reward_plot):
    plt.figure(2)  # ??
    plt.clf()  # ??
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('TestsetRewards')
    plt.plot(episode_durations, ave_reward_plot)
    plt.pause(0.001)

def main():
    # init a task generator for data fetching
    task = gym.make("InjectWorld-v1")
    init_state = task.reset()
    STATE_DIM = task.observation_space.shape[0]
    ACTION_DIM = task.action_space.shape[0]
    STEP = 2000
    SAMPLE_NUMS = 8
    ACTION_LIM = task.action_space.high[0]
    #init noise
    noise = utils.OrnsteinUhlenbeckActionNoise(ACTION_DIM)
    # init value network
    value_network = ValueNetwork(input_size = STATE_DIM ,hidden_size=40, output_size=1)
    value_network_optim = torch.optim.Adam(value_network.parameters(), lr=0.01)

    # init actor network
    actor_network = ActorNetwork(STATE_DIM, 40, ACTION_DIM, ACTION_LIM)
    actor_network_optim = torch.optim.Adam(actor_network.parameters(), lr =0.01)

    steps = []
    task_episodes  =[]
    test_results = []
    episode_durations = []
    for step in range(STEP):
        states, actions, rewards, final_r, current_state = roll_out(actor_network,task, SAMPLE_NUMS,value_network, init_state, noise,ACTION_LIM)
        init_state = current_state
        actions_var = Variable(torch.Tensor(actions).view(-1, ACTION_DIM))
        states_var = Variable(torch.Tensor(states).view(-1, STATE_DIM))

        #train actor network
        actor_network_optim.zero_grad()
        tanh_action =actor_network(states_var)
        vs = value_network(states_var).detach()#detach from the compute graph
        qs = Variable(torch.Tensor(discount_reward(rewards, 0.99, final_r)))
        advantages = qs - vs
        #actor_network_loss = - torch.mean(torch.sum(tanh_action*actions_var,1)*advantages)
        actor_network_loss =torch.mean(torch.sum(tanh_action*actions_var,1)*advantages)
        actor_network_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_network.parameters(),0.5)
        actor_network_optim.step()

        #train value network
        value_network_optim.zero_grad()
        target_values = qs
        values = value_network(states_var)
        criterion = nn.MSELoss()
        value_network_loss= criterion(values, target_values)
        value_network_loss.backward()
        torch.nn.utils.clip_grad_norm(value_network.parameters(), 0.5)
        value_network_optim.step()

        # Testing
        if (step + 1)% 100 == 0:
            result = 0
            test_task = gym.make("InjectWorld-v1")
            for test_epi in range(10):
                state = test_task.reset()
                for test_step in range(10):
                    tanh_action = actor_network(Variable(torch.Tensor([state])))
                    action = np.multiply(tanh_action.data.numpy()[
                                         0], [20, 20, 20])
                    next_state, reward, done, _ = test_task.step(action)
                    result += reward
                    state =next_state
                    if done:
                        break
            print("step", step +1, "test result", result/10)
            steps.append(step+1)
            test_results.append(result/10)
            episode_durations.append(step + 1)
            plot_durations(plt, episode_durations, test_results)


if __name__ == '__main__':
    main()
