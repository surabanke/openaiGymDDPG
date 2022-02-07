#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:46:17 2021

@author: eunhwalee
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



env = gym.make('MountainCarContinuous-v0').unwrapped
observation_space = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
# set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',('state', 'action', 'reward', 'next_state'))

class replaymemory(object):
    
    def __init__(self,buffer_capacity):
        self.buffer = deque([], maxlen = buffer_capacity)
        
    def push(self,*T):
      
        self.buffer.append(Transition(*T))
        
    def minibatch(self,minibatch_size):
        
        return random.sample(self.buffer,minibatch_size)  # memory.minibatch is array 
        
    
class Qnetwork(nn.Module):
    def __init__(self,observation_space,action_space):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(observation_space + action_space, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, 20), 
            nn.ReLU(), 
            nn.Linear(20, action_space)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat((state, action), 1))

    #     self.critic_FCobs1 = nn.Linear(observation_space,100)
    #     self.critic_FCobs2 = nn.Linear(100,50)
    #     self.critic_FCobs3 = nn.Linear(50,action_space)


    # def forward_Critic(self,obs):
    #     obs.requires_grad = False

    #     obs = self.critic_FCobs1(obs)
    #     obs = F.relu(self.critic_FCobs2(obs))
    #     obs = self.critic_FCobs3(obs)
        
    #     return obs

    
    
# Actor
class Actor(nn.Module):
    def __init__(self,observation_space):
        super().__init__()
        #self.scale = torch.Tensor(2)
        self.layer1 = nn.Linear(observation_space,100)
        self.layer2 = nn.Linear(100,100)
        self.layer3 = nn.Linear(100,50)
        self.layer4 = nn.Linear(50,1)
        
        
    def forward(self,s):
        s.requires_grad = False
        #s = s.to(device)
        res = F.relu(self.layer1(s)) 
        res = F.relu(self.layer2(res)) 
        res = F.relu(self.layer3(res)) 
        res = torch.tanh(self.layer4(res)) 
        # no need to scale it
        
        return res
        
    
#### Get the information of environment

def get_action(state):
    
    action = mainactor_net(state).cpu().data.numpy().squeeze(0)
    
    return action


# def get_screen():
#     screen = env.render(mode='rgb_array').transpose((2, 0, 1)) #env.render()
#     _, screen_height, screen_width = screen.shape
#     dim = screen.shape[0]
#     screen_height = screen.shape[1]
#     screen_width = screen.shape[2]
#     env.observation_space
    

env.reset()
plt.figure()
#plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()


### Training

buffer_capacity = 100000
minibatch_size = 100

episodes = 1000
discount_factor = 0.99
target_update_freq = 100
eps_init = 0.9
eps_min = 0.05
eps_decayRate = 200
memory = replaymemory(buffer_capacity)



maincritic_net = Qnetwork(observation_space,action_space).to(device)
mainactor_net = Actor(observation_space).to(device)

targetcritic_net = Qnetwork(observation_space,action_space).to(device)
targetactor_net = Actor(observation_space).to(device)
optimizer = optim.Adam(maincritic_net.parameters(), lr = 1e-3)
done = False

def optimize_model(Data):
    if len(memory) < minibatch_size:
        return
    minibatchData = memory.minibatch(minibatch_size)
    Data = Transition(*zip(*minibatchData))  # zip unpacking -> *zip
    
    #Data =  torch.tensor()
    batchstate = torch.cat(Data.state)  ##################################################################### !!
    batchaction = torch.cat(Data.action)
    batchreward = torch.cat(Data.reward)
    batch_next = torch.cat(Data.next_state)
    next_action = targetactor_net(batch_next)
    
    
    Q_values = maincritic_net(batchstate,batchaction)
    
    
    target_Q_values = targetcritic_net(batch_next,next_action)
    target_Q_values = (target_Q_values * discount_factor) + batchreward
        
    criterion = nn.SmoothL1Loss()
    Q_loss = criterion(Q_values, target_Q_values.unsqueeze(1))
    
    actor_loss = - (maincritic_net(batchstate).gather(1,mainactor_net(batchstate)))
    actor_loss = np.mean(actor_loss)

    

    optimizer.zero_grad()
    Q_loss.backward()
    actor_loss.backward()
    # gradient clipping for all parameters in layers
    for param in maincritic_net.parameters():
        param.grad.data.clamp_(-1, 1) 
    optimizer.step()

    
    
while not done:
    
    state = env.reset()    
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    action = get_action(state)
    action = torch.from_numpy(action).unsqueeze(0)
    next_state,reward,done,info = env.step(action)
    next_state = torch.from_numpy(next_state).float().unsqueeze(0)
    reward = torch.tensor([reward])
    
    # a = next_state[0]
    # b = next_state[1]
    # arr = [a.tolist(), b.tolist()]
    # next_state = torch.Tensor(arr)
    # next_state = next_state[1]
    
    memory.push(state,action,reward,next_state)
    
    state = next_state 
    
    optimize_model(Data)
    




    

















