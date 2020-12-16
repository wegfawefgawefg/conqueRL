import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from model import DQNetwork
from replayBuffer import ReplayBuffer

class DQAgent():
    def __init__(self, lr, inputChannels, stateShape, numActions, batchSize, 
            epsilon=1.0, gamma=0.99, layer1Size=1024, layer2Size=512, 
            maxMemSize=100000, epsMin=0.01, epsDecay=5e-4):
        self.lr = lr
        self.epsilon = epsilon
        self.epsMin = epsMin
        self.epsDecay = epsDecay
        self.gamma = gamma
        self.batchSize = batchSize
        self.actionSpace = list(range(numActions))
        self.maxMemSize = maxMemSize
        
        self.memory = ReplayBuffer(maxMemSize, stateShape)
        self.deepQNetwork = DQNetwork(lr, inputChannels, numActions)

    '''
    REENABLE EPSILON GREEDY
    '''
    def chooseAction(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation).float().clone().detach()
            state = state.to(self.deepQNetwork.device)
            state = state.unsqueeze(0)
            policy = self.deepQNetwork(state)
            action = torch.argmax(policy).item()
            return action
        else:
            return np.random.choice(self.actionSpace)

    def storeMemory(self, state, action, reward, nextState, done):
        self.memory.storeMemory(state, action, reward, nextState, done)

    def learn(self):
        if self.memory.memCount < self.batchSize:
            return

        self.deepQNetwork.optimizer.zero_grad()
    
        stateBatch, actionBatch, rewardBatch, nextStateBatch, doneBatch = \
            self.memory.sample(self.batchSize)
        stateBatch = torch.tensor(stateBatch).to(self.deepQNetwork.device)
        actionBatch = torch.tensor(actionBatch).to(self.deepQNetwork.device)
        rewardBatch = torch.tensor(rewardBatch).to(self.deepQNetwork.device)
        nextStateBatch = torch.tensor(nextStateBatch).to(self.deepQNetwork.device)
        doneBatch = torch.tensor(doneBatch).to(self.deepQNetwork.device)
        
        batchIndex = np.arange(self.batchSize, dtype=np.int64)

        actionQs = self.deepQNetwork(stateBatch)[batchIndex, actionBatch]
        allNextActionQs = self.deepQNetwork(nextStateBatch)
        nextActionQs = torch.max(allNextActionQs, dim=1)[0]
        nextActionQs[doneBatch] = 0.0
        qTarget = rewardBatch + self.gamma * nextActionQs

        loss = self.deepQNetwork.loss(qTarget, actionQs).to(self.deepQNetwork.device)
        loss.backward()
        self.deepQNetwork.optimizer.step()

        if self.epsilon > self.epsMin:
            self.epsilon -= self.epsDecay

if __name__ == "__main__":
    numStackedFrames = 4
    inputShape = (1, numStackedFrames, 64, 64)
    stateShape = (numStackedFrames, 64, 64)

    #   make agent
    agent = DQAgent(lr=0.001, inputChannels=4, stateShape=stateShape, numActions=3, batchSize=1)

    #   make fake data
    x = torch.ones(inputShape)
    xIn = x.to(agent.deepQNetwork.device)
    print(x.shape)
    
    #   compute some actions
    action = agent.chooseAction(xIn)
    print("action {}".format(action))

    #   test learn
    for i in range(agent.batchSize):
        agent.storeMemory(x, action, 0, x, False)
    agent.learn()

    print("TEST DONE")

