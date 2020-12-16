import math
import random
import copy

import gym       
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

class ICM(torch.nn.Module):
    '''ICM module as per "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363
    '''
    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

        self.hidden_size = 64
        self.state_encoding_size = 64

        self.state_encoder = nn.Sequential(
            nn.Linear(*state_shape,     self.hidden_size),  nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),  nn.ReLU(),
            nn.Linear(self.hidden_size, self.state_encoding_size))

        self.pred_retrospective_action = nn.Sequential(
            nn.Linear(self.state_encoding_size * 2, self.hidden_size),  nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),              nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_actions))

        self.pred_next_state_encoding = nn.Sequential(
            nn.Linear(self.state_encoding_size + self.num_actions, self.hidden_size),   nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),                              nn.ReLU(),
            nn.Linear(self.hidden_size, self.state_encoding_size))

    def forward(self, states, next_states, actions):
        state_encodings      = self.state_encoder(states)
        next_state_encodings = self.state_encoder(next_states)

        state_and_next_state_encodings = torch.cat([state_encodings, next_state_encodings], dim=1)
        retrospective_action_preds = self.pred_retrospective_action(state_and_next_state_encodings)

        state_and_action_encodings = torch.cat([state_encodings, actions], dim=1)
        next_state_encoding_preds = self.pred_next_state_encoding(state_and_action_encodings)

        return next_state_encodings, next_state_encoding_preds, retrospective_action_preds

class QNetwork(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.fc1Dims = 128
        self.fc2Dims = 64

        self.q_values = nn.Sequential(
            nn.Linear(*state_shape,  self.fc1Dims), nn.ReLU(),
            nn.Linear( self.fc1Dims, self.fc2Dims), nn.ReLU(),
            nn.Linear( self.fc2Dims, num_actions ))

    def forward(self, x):
        return self.q_values(x)

class LinearSchedule:
    def __init__(self, start, end, num_steps):
        self.delta = (end - start) / float(num_steps)
        self.num = start - self.delta
        self.count = 0
        self.num_steps = num_steps

    def value(self):
        return self.num

    def step(self):
        if self.count <= self.num_steps:
            self.num += self.delta
        self.count += 1

        return self.num

class Agent():
    def __init__(self, 
            state_shape,
            num_actions,
            batch_size=32,
            gamma=0.99,
            learn_rate=3e-4,    
            buffer_size=100_000,
            min_buffer_fullness=64,

            icm_max_reward=1.0,
            no_extrinsic_rewards=False,
            ):

        '''     SETTINGS    '''
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
        # self.device = torch.device("cpu")

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.gamma = gamma
        self.icm_max_reward = icm_max_reward
        self.no_extrinsic_rewards=no_extrinsic_rewards

        self.min_buffer_fullness = min_buffer_fullness

        self.net_copy_interval = 10

        '''     STATE       '''
        self.learn_step_counter = 0
        self.memory_minimum_fullness_announced = False

        self.memory = ReplayBuffer(size=1_000_000, state_shape=state_shape, num_actions=self.num_actions)

        self.epsilon = LinearSchedule(start=1.0, end=0.01, num_steps=500)

        self.q_net = QNetwork(state_shape, num_actions).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net).to(self.device)
        self.icm = ICM(state_shape, num_actions).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) +  list(self.icm.parameters()), lr=learn_rate)

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

    def get_icm_loss(self, data):
        states, actions, rewards, states_, dones = data
        next_state_enc, next_state_enc_preds, retrospective_action_preds = self.icm(states, states_, actions)

        forward_loss = (next_state_enc - next_state_enc_preds).sum(dim=1)**2
        inverse_loss = (actions - retrospective_action_preds).sum(dim=1)**2

        return forward_loss, inverse_loss

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
        forward_loss, inverse_loss = self.get_icm_loss((states, actions, rewards, states_, dones))
        intrinsic_rewards = torch.clamp(forward_loss, 0, self.icm_max_reward).detach().view(self.batch_size, 1)
        if self.no_extrinsic_rewards:
            rewards = intrinsic_rewards
        else:
            rewards += intrinsic_rewards
        q_network_loss = self.get_q_network_loss((states, actions, rewards, states_, dones))

        forward_loss = forward_loss.mean()
        inverse_loss = inverse_loss.mean()

        self.optimizer.zero_grad()
        loss = q_network_loss + forward_loss + inverse_loss
        loss.backward()
        self.optimizer.step()

        self.epsilon.step()

        if self.learn_step_counter % self.net_copy_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.learn_step_counter += 1

import skimage.measure

def squash_state(state):
    state = state.mean(axis=2)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = state.flatten()
    return state

SPEED = 0.2

def convert_actions(action, env_actions):
    gas, steer, brake = env_actions[0], env_actions[1], env_actions[2]

    if action == 0:     #   gas up
        gas += SPEED
    elif action == 1:   #   gas down
        gas -= SPEED
    elif action == 2:   #   steer up
        steer += SPEED
    elif action == 3:   #   steer down
        steer -= SPEED
    elif action == 4:   #   break on
        brake = SPEED
    elif action == 5:   #   break off
        brake = 0.0

    # if action == 0:     #   forward
    #     gas = 1.0
    # elif action == 1:   #   forward left
    #     gas = 1.0
    #     steer = 1.0
    # elif action == 2:   #   forward right
    #     gas = 1.0
    #     steer = -1.0
    # # elif action == 3:   #   break
    # #     brake = 1.0

    env_actions[0] = steer
    env_actions[1] = gas
    env_actions[2] = brake
    
    env_actions.clip(-1.0, 1.0)

    return env_actions

def stack_frames(frame_stack):
    stack = np.concatenate(list(frame_stack))
    return stack

from collections import deque

REWARD_MIN = 5.0
FRAME_BUFFER_SIZE = 4

if __name__ == '__main__':
    env = gym.make('CarRacing-v0').unwrapped
    STATE_SHAPE = (12**2 * FRAME_BUFFER_SIZE,)
    agent = Agent(state_shape=STATE_SHAPE, num_actions=4)

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:

        done = False
        state = env.reset()
        state = squash_state(state)

        frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        for i in range(FRAME_BUFFER_SIZE):
            frame_buffer.append(state)
        state = stack_frames(frame_buffer)

        score, frame = 0, 1
        peak_env_reward = 0
        env_actions = np.zeros(3, dtype=np.float)
        while not done:
            env.render()

            action = agent.choose_action(state)
            env_actions = convert_actions(action, env_actions)
            state_, reward, done, info = env.step(env_actions)
            state_ = squash_state(state_)
            frame_buffer.append(state_)
            state_ = stack_frames(frame_buffer)

            reward = max(reward, 0.0)
            if env.reward < (peak_env_reward - REWARD_MIN) \
                    or env.reward < 0:
                print("track failed")
                done = True

            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            num_samples += 1
            score += reward
            frame += 1

            peak_env_reward = max(peak_env_reward, env.reward)


        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}, frame {:12.3f}, eps {:12.3f}").format(
            num_samples, episode, high_score, score, frame, agent.epsilon.value()))

        episode += 1