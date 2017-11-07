import gym
import numpy as np
import random 
from collections import deque

import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
#------------------
#Class DQN
#Hyper Parameters of DQN
# Hyper Parameters for DQN


FloatTensor =  torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor
class DQN(nn.Module):
	"""Agent of DQN"""
	def __init__(self, env):
		
		super(DQN, self).__init__()
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		self.fc1 = nn.Linear(self.state_dim, 20)
		self.fc2 = nn.Linear(20, self.action_dim)

	def forward(self, x):
		#forward compute
		x = F.relu(self.fc1(x))
		return self.fc2(x)

class Agent():
	"""docstring for Agent"""
	def __init__(self,env):
		# experience replay
		self.replay_buffer = deque()
		self.time_step = 0
		self.epsilon = EPS_START
		self.state_dim = env.observation_space.shape[0]
		self.action_dim = env.action_space.n
		
	def perceive(self, state, action, reward, next_state,done,Net):
		one_hot_action = np.zeros(self.action_dim)
		one_hot_action[action] = 1
		self.replay_buffer.append((state, one_hot_action, reward, next_state,done))
		if len(self.replay_buffer) > REPLAY_SIZE:
			self.replay_buffer.popleft()
		if len(self.replay_buffer) > BATCH_SIZE:
			self.train_Q_network(Net)

	def train_Q_network(self,Net):
		self.time_step += 1
		optimizer = optim.Adam(Net.parameters())
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
		state_batch = Variable(translate(np.array([data[0] for data in minibatch])))
		action_batch = Variable(translate_index(np.array([data[1] for data in minibatch])))
		reward_batch = Variable(translate(np.array([data[2] for data in minibatch])))
		next_state_batch = Variable(translate(np.array([data[3] for data in minibatch])))
		 # Step 2: calculate y
		target_y_batch = []
		Q_value_batch = torch.sum(Net(state_batch)*action_batch.type(torch.FloatTensor),1)
		

		next_Q_value_batch = Net(next_state_batch)

		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				target_y_batch.append((reward_batch[i].data).numpy()[0])
			else :
				target_y_batch.append((reward_batch[i].data + GAMMA * torch.max(next_Q_value_batch[i].data)).numpy()[0])
		#print(target_y_batch)	
		loss = F.mse_loss(Variable(translate(np.array(target_y_batch))), Q_value_batch)
	    # Optimize the model
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	def egreedy_action(self, state,Net):
		
		sample = random.random()
		if sample <= self.epsilon:
			return random.randint(0,self.action_dim-1)
		else:
			#print(Net(Variable(translate(state))).data.max(0)[1][0])
			return Net(Variable(translate(state))).data.max(0)[1][0]
		self.epsilon -= (EPS_START - EPS_END)/10000
	def action(self,state,Net):
		return Net(Variable(translate(state))).data.max(0)[1][0]
def translate(x):
	return torch.from_numpy(x).type(FloatTensor)
def translate_index(x):
	return torch.from_numpy(x).type(LongTensor)

ENV_NAME = "CartPole-v0"
EPISODE = 10000
STEP = 300
TEST = 10 
GAMMA = 0.99 # discount factor for target Q
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.05 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
                # 

def main():
	env = gym.make(ENV_NAME)
	model = DQN(env)
	agent = Agent(env)
	for episode in range(EPISODE):
		#initialize task
		print('#### {} ####'.format(episode))
		state = env.reset()
		#Train 
		for step in range(STEP):
			action = agent.egreedy_action(state,model) # e-greedy action for train
			#print(action,action[0])
			next_state, reward,done,_ = env.step(action)
			#Define reward for agent
			reward_agent = -1 if done else 0.1
			agent.perceive(state,action,reward,next_state, done,model)
			state = next_state
			if done:
				break
		#Test every 100 episodes:
		if episode % 100 == 0:
			total_reward = 0
			for i in range(TEST):
				state = env.reset()
				for j in range(STEP):
					env.render()
					action = agent.action(state,model)
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print('episode:{},Evaluation Average Reward:{}'.format(episode, ave_reward))	
			if ave_reward >= 200:
				break

if __name__ == '__main__':
	main()