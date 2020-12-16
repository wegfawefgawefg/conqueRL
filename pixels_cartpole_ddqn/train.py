import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym       
import math
from matplotlib import pyplot as plt
from skimage import transform
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np
import os
import datetime

from agent import DQAgent

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def formatState(state):
    state = state.mean(axis=0)

    # state = np.mean(state, -1)
    state = state[30:-20, 30:-30]
    state = state/255.0
    state = transform.resize(state, [64,64])
    # plt.imshow(state, cmap='gray', vmin=0, vmax=1.0)
    # plt.show()

    return state

def makeFrameStack(frameBuffer, NUM_SKIP_BTWN_FRAMES):
    frameStackList = list(frameBuffer)[::-(NUM_SKIP_BTWN_FRAMES+1)]
    observation = np.stack(frameStackList)
    # observation = observation.reshape(1, *observation.shape)
    # observation = torch.tensor(observation.copy()).float()
    # observation = observation.to(agent.deepQNetwork.device)
    # observation = observation.unsqueeze(0)

    return observation

if __name__ == '__main__':
    CHECKPOINT_INTERVAL = 1000
    ENV_NAME = "Cartpole"
    DISC_OR_CONT = "Disc_Pix"
    ALGO_NAME = "DQN"
    LR = 0.00001

    #   TENSORBOARD BOOK-KEEPING
    YMDHMS = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    CHECKPOINT_NAME = "_".join([ENV_NAME, DISC_OR_CONT, ALGO_NAME, str(LR)])
    RUN_NAME = "_".join([YMDHMS, CHECKPOINT_NAME])
    RUNS_PATH = os.path.join("..", "runs", RUN_NAME)
    writer = SummaryWriter(RUNS_PATH, comment=CHECKPOINT_NAME)

    #   MAKE ENV
    env = gym.make('CartPole-v1').unwrapped

    CHECKPOINT_INTERVAL = 100

    #   SETUP FRAME BUFFER
    '''-currently using a non stack overlapping buffer
        -should experiment with a strictly overlapping one, and compare perf'''
    NUM_SKIP_BTWN_FRAMES = 1
    FRAME_STACK_SIZE = 4
    FRAME_BUFFER_SIZE = FRAME_STACK_SIZE * (NUM_SKIP_BTWN_FRAMES + 1)

    #   make agent
    stateShape = (FRAME_STACK_SIZE, 64, 64)

    #   make agent
    #   #   lr above 0.0001 wont necessarily converge (tested somewhat)
    agent = DQAgent(lr=LR, inputChannels=4, stateShape=stateShape, 
        numActions=2, batchSize=64)

    #   STATS
    scoreHistory = []
    numHiddenEpisodes = -1
    highScore = -math.inf
    recordTimeSteps = math.inf

    episode = 0
    while True:
        frameBuffer = deque(maxlen=FRAME_BUFFER_SIZE)
        
        #   PREP GAME STATE
        done = False
        env.reset()
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))
        state = formatState(screen)

        for i in range(FRAME_BUFFER_SIZE):
            frameBuffer.append(state)
        observation = makeFrameStack(frameBuffer, NUM_SKIP_BTWN_FRAMES)

        score, frame = 0, 1
        while not done:
            # if episode > numHiddenEpisodes:
            #     env.render()

            actionNum = agent.chooseAction(observation)

            state_, reward, done, info = env.step(actionNum)
            screen = env.render(mode='rgb_array').transpose((2, 0, 1))
            state_ = formatState(screen)

            frameBuffer.append(state_)

            nextObservation = makeFrameStack(frameBuffer, NUM_SKIP_BTWN_FRAMES)

            agent.storeMemory(observation, actionNum, reward, nextObservation, done)
            agent.learn()

            observation = nextObservation

            score += reward
            frame += 1

        scoreHistory.append(score)

        recordTimeSteps = min(recordTimeSteps, frame)
        highScore = max(highScore, score)
        
        writer.add_scalar("score", score, episode)
        writer.add_scalar("shortestBalanceTime", recordTimeSteps, episode)
        writer.add_scalar("highScore", highScore, episode)

        print(( "ep {}: high-score {:12.3f}, shortest-time {:d}, "
                "score {:12.3f}, last-epidode-time {:4d}").format(
            episode, 
            highScore, 
            recordTimeSteps, 
            score,
            frame,
            ))

        #   model checkpoints
        if episode % CHECKPOINT_INTERVAL == 0:
            name = "checkpoint_%d_%d.dat" % (episode, score)
            fname = os.path.join('.', 'checkpoints', name)
            torch.save( agent.deepQNetwork.state_dict(), fname )

        episode += 1
