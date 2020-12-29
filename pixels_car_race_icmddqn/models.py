import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ConvFrontEnd(torch.nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.num_channels = input_shape[0]

        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.maxPool1 = nn.MaxPool2d(2, 2)
        # self.maxPool2 = nn.MaxPool2d(2, 2)
        # self.maxPool3 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.maxPool1(x)
        x = F.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.maxPool2(x)
        x = F.relu(x)

        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.maxPool3(x)
        x = F.relu(x)

        return x

class FCThree(torch.nn.Module):
    def __init__(self, input_shape, num_outputs):
        super().__init__()
        self.fc1Dims = 128
        self.fc2Dims = 64

        self.fc = nn.Sequential(
            nn.Linear( input_shape,  self.fc1Dims), nn.ReLU(),
            nn.Linear( self.fc1Dims, self.fc2Dims), nn.ReLU(),
            nn.Linear( self.fc2Dims, num_outputs))

    def forward(self, x):
        return self.fc(x)

class ThreeByThree(torch.nn.Module):
    def __init__(self, input_shape, num_outputs):
        super().__init__()
        self.conv_out_shape = 64 * 2 * 2

        self.conv_front_end = ConvFrontEnd(input_shape)
        self.fc_q = FCThree(
            input_shape=self.conv_out_shape, 
            num_outputs=num_outputs)

    def forward(self, x):
        batchSize = x.shape[0]
        x = self.conv_front_end(x)
        # print("conv out shape: {}".format(x.shape))
        # quit()
        x = x.view(batchSize, self.conv_out_shape)
        x = self.fc_q(x)

        return x

class ICM(torch.nn.Module):
    '''ICM module as per "Curiosity-driven Exploration by Self-supervised Prediction"
    https://arxiv.org/abs/1705.05363
    '''
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.state_encoding_size = 64

        self.state_encoder = ThreeByThree(
            input_shape=input_shape, 
            num_outputs=self.state_encoding_size)
        self.pred_retrospective_action = FCThree(
            input_shape=self.state_encoding_size * 2, 
            num_outputs=self.num_actions)
        self.pred_next_state_encoding = FCThree(
            input_shape=self.state_encoding_size + self.num_actions, 
            num_outputs=self.state_encoding_size)

    def forward(self, states, next_states, actions):
        state_encodings      = self.state_encoder(states)
        next_state_encodings = self.state_encoder(next_states)

        state_and_next_state_encodings = torch.cat([state_encodings, next_state_encodings], dim=1)
        retrospective_action_preds = self.pred_retrospective_action(state_and_next_state_encodings)

        state_and_action_encodings = torch.cat([state_encodings, actions], dim=1)
        next_state_encoding_preds = self.pred_next_state_encoding(state_and_action_encodings)

        return next_state_encodings, next_state_encoding_preds, retrospective_action_preds

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu,")
    # device = torch.device("cpu")

    FRAME_STACK_SIZE = 4
    STATE_SHAPE = (FRAME_STACK_SIZE, 64, 64)
    INPUT_SHAPE = (1, *STATE_SHAPE)

    net = ConvQNetwork(state_shape=STATE_SHAPE, num_actions=3).to(device)

    #   make fake data
    x = torch.ones(INPUT_SHAPE).to(device)
    print(x.shape)

    #   feedforward 
    qValues = net(x)
    print("qValues {}".format(qValues))