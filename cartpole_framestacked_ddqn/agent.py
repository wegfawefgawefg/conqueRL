import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from replay_buffer import ReplayBuffer
from models import FCQNetwork
from utils import LinearSchedule

class Agent():
    def __init__(self, 
            state_shape,
            num_actions,
            batch_size=64,
            gamma=0.99,
            learn_rate=3e-4,    
            buffer_size=100_000,
            min_buffer_fullness=64,
            ):

        '''     SETTINGS    '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
        # self.device = torch.device("cpu")

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma

        self.min_buffer_fullness = min_buffer_fullness

        self.net_copy_interval = 10

        '''     STATE       '''
        self.learn_step_counter = 0
        self.memory_minimum_fullness_announced = False

        self.memory = ReplayBuffer(size=1_000_000, state_shape=state_shape, num_actions=self.num_actions)

        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=500)

        self.q_net = FCQNetwork(state_shape, num_actions).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learn_rate)

    def choose_action(self, observation):
        if random.random() > self.epsilon.value():
            state = torch.tensor(observation).float().detach()
            state = state.to(self.device)
            state = state.unsqueeze(0)

            q_values = self.q_net(state)
            action = torch.argmax(q_values).item()

            return action
        else:
            action = random.randint(0, self.num_actions - 1)
            return action

    def store_memory(self, state, action, reward, state_, done):
        #   one hot encode the action for discrete action space
        action_one_hot = np.zeros(self.num_actions, dtype=np.float32)
        action_one_hot[action] = 1.0

        self.memory.store_memory(state, action_one_hot, reward, state_, done)

    def get_q_network_loss(self, data):
        states, actions, rewards, states_, dones = data

        batch_indices = np.arange(self.batch_size, dtype=np.int64)
        chosen_actions = torch.max(actions, dim=1)[1]                                           #   (batch_size, num_actions)
        action_qs = self.q_net(states)[batch_indices, chosen_actions].view(self.batch_size, 1)  #   (batch_size, 1)

        qs_ = self.target_q_net(states_)                                    #   (batch_size, num_actions)
        policy_qs = self.q_net(states_)                                     #   (batch_size, num_actions)
        actions_ = torch.max(policy_qs, dim=1)[1]                           #   (batch_size)
        action_qs_ = qs_[batch_indices, actions_].view(self.batch_size, 1)  #   (batch_size, 1)
        action_qs_[dones] = 0.0

        q_targets = rewards + self.gamma * action_qs_

        loss = F.mse_loss(q_targets, action_qs)

        return loss

    def learn(self):
        '''     learn function scheduler    '''
        if self.memory.count < self.min_buffer_fullness:
            return
        elif not self.memory_minimum_fullness_announced:
            print("Memory reached minimum fullness.")
            self.memory_minimum_fullness_announced = True

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size, self.device)
        q_network_loss = self.get_q_network_loss((states, actions, rewards, states_, dones))

        self.optimizer.zero_grad()
        loss = q_network_loss
        loss.backward()
        self.optimizer.step()

        self.epsilon.step()

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.learn_step_counter += 1