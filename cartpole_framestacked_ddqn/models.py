import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class FCQNetwork(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.fc1Dims = 512
        self.fc2Dims = 256

        self.q_values = nn.Sequential(
            nn.Linear(*state_shape,  self.fc1Dims), nn.ReLU(),
            nn.Linear( self.fc1Dims, self.fc2Dims), nn.ReLU(),
            nn.Linear( self.fc2Dims, num_actions ))

    def forward(self, x):
        return self.q_values(x)


class ConvQNetwork(torch.nn.Module):
    def __init__(self, state_shape, num_actions):
        super().__init__()
        self.num_channels = state_shape[1]
        self.convOutShape = 32 * 5 * 5
        self.fc1Dims = 1024
        self.fc2Dims = 512

        self.conv1 = nn.Conv2d(self.num_channels, 16, 5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)

        # self.maxPool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.convOutShape, self.fc1Dims)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)
        self.fc3 = nn.Linear(self.fc2Dims, num_actions)

    def forward(self, x):
        batchSize = x.shape[0]
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.maxPool1(x)
        # print("post conv maxpool shape: {}".format(x.shape))

        x = x.view(batchSize, self.convOutShape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        qValues = self.fc3(x)
        return qValues

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
    # device = torch.device("cpu")

    FRAME_STACK_SIZE = 4
    INPUT_SHAPE = (1, FRAME_STACK_SIZE, 64, 64)

    net = ConvQNetwork(state_shape=INPUT_SHAPE, num_actions=3).to(device)

    #   make fake data
    x = torch.ones(INPUT_SHAPE).to(device)
    print(x.shape)

    #   feedforward 
    qValues = net(x)
    print("qValues {}".format(qValues))