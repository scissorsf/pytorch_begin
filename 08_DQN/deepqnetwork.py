import gym
import numpy as np
import random 
from collections import deque
import matplotlib.pyplot as plt

import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
#------------------
#Class DQN

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
		
#------------------
#Hyper Parameters
ENV_NAME = "CartPole-v0"
EPISODE = 10000
STEP = 300
TEST = 10 
GAMMA = 0.99 # discount factor for target Q
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.05 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch


def main():
	#initialize OpenAI Gym env and dqn agent
	env = gym.make(ENV_NAME)
	STATE_DIM = env.observation_space.shape[0]
	ACTION_DIM = env.action_space.n
	replay_buffer = deque()
	model = DQN(env)
	criterion = nn.MSELoss()
	optimizer = optim.RMSprop(model.parameters())

	episode_durations = []
	ave_reward_plot = []
	#agent = Agent(env)
	def translate(x):
		return torch.from_numpy(x).type(FloatTensor)
	def translate_index(x):
		return torch.from_numpy(x).type(LongTensor)
	
	def egreedy_action(state):
		translate(state)
		sample = random.random()
		eps_threshold = EPS_START
		if sample <= eps_threshold:
			return LongTensor([[random.randint(0,ACTION_DIM-1)]])# size of 1x1
		else:
			
			return model(Variable(translate(state), volatile=True).type(FloatTensor)).data.max(0)[1].view(1,1)
		eps_threshold -= (EPS_START - EPS_END)/10000

	def do_action(state):
		return  model(Variable(translate(state), volatile=True).type(FloatTensor)).data.max(0)[1].view(1,1)
	def percieve(state, action, reward, next_state,done):
		replay_buffer.append((state, action, reward, next_state,done))# 除了done意外全是Tensor
		if len(replay_buffer) > REPLAY_SIZE:
			replay_buffer.popleft()


	def plot_durations():
		plt.figure(2)#??
		plt.clf()#??
		plt.title('Training...')
		plt.xlabel('Episode')
		plt.ylabel('Duration')
		plt.plot(episode_durations,ave_reward_plot)
		plt.pause(0.001)  # pause a bit so that plots are updated		
		                  #
	def optimize_model():
		if len(replay_buffer) < BATCH_SIZE:
			return
		minibatch = random.sample(replay_buffer,BATCH_SIZE)
		state_batch = Variable(translate(np.array([data[0] for data in minibatch])))
		action_batch = Variable(torch.cat([data[1] for data in minibatch]))
		reward_batch = Variable(translate(np.array([data[2] for data in minibatch])))
		next_state_batch = Variable(translate(np.array([data[3] for data in minibatch])))
		 # Step 2: calculate y
		target_y_batch = []
		Q_value_batch = model(state_batch).gather(1,action_batch)
		#print(Q_value_batch[:3])

		next_Q_value_batch = model(next_state_batch)

		for i in range(0,BATCH_SIZE):
			done = minibatch[i][4]
			if done:
				target_y_batch.append((reward_batch[i].data).numpy()[0])
			else :
				target_y_batch.append((reward_batch[i].data + GAMMA * torch.max(next_Q_value_batch[i].data)).numpy()[0])
		#print(target_y_batch)	
		loss = criterion(Q_value_batch,Variable(translate(np.array(target_y_batch)), volatile=False))
	    # Optimize the model
		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

	for episode in range(EPISODE):
		#initialize task
		#print('#### {} ####'.format(episode))
		state = env.reset()
		#Train 
		for step in range(STEP):
			action = egreedy_action(state) # e-greedy action for train
			#print(action)
			                               # 
			#print(action,action[0])
			next_state, reward,done,_ = env.step(action[0,0])
			#Define reward for agent
			#reward_agent = -1 if done else 0.1
			percieve(state,action,reward,next_state, done)
			state = next_state
			optimize_model()
			if done:
				break
		#Test every 100 episodes:
		if episode % 100 == 0:
			total_reward = 0
			for i in range(TEST):
				state = env.reset()
				for j in range(STEP):
					env.render()
					action = do_action(state)
					state,reward,done,_ = env.step(action[0,0])
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print('episode:{},Evaluation Average Reward:{}'.format(episode, ave_reward))
			episode_durations.append(episode)
			ave_reward_plot.append(ave_reward)	
			plot_durations()
			print(episode_durations, ave_reward_plot)
			if ave_reward >= 200:
				break


if __name__ == '__main__':
	main()
	
