import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer:
    def __init__(self, size, state_shape, num_actions):
        self.size = size
        self.count = 0

        self.state_memory       = np.zeros((self.size, *state_shape ), dtype=np.float32)
        self.action_memory      = np.zeros((self.size, num_actions  ), dtype=np.float32)
        self.reward_memory      = np.zeros((self.size, 1            ), dtype=np.float32)
        self.next_state_memory  = np.zeros((self.size, *state_shape ), dtype=np.float32)
        self.done_memory        = np.zeros((self.size, 1            ), dtype=np.bool   )

    def store_memory(self, state, action, reward, next_state, done):
        index = self.count % self.size 

        self.state_memory[index]      = state
        self.action_memory[index]     = action
        self.reward_memory[index]     = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index]       = done

        self.count += 1

    def sample(self, sample_size, device):
        highest_index = min(self.count, self.size)
        indices = np.random.choice(highest_index, sample_size, replace=False)

        states  = self.state_memory[indices]
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        states_ = self.next_state_memory[indices]
        dones   = self.done_memory[indices]

        states  = torch.tensor( states  ).to(device)
        actions = torch.tensor( actions ).to(device)
        rewards = torch.tensor( rewards ).to(device)
        states_ = torch.tensor( states_ ).to(device)
        dones   = torch.tensor( dones   ).to(device)

        return states, actions, rewards, states_, dones