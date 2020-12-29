import math

import numpy as np
import gym       
import skimage.measure

from agent import Agent
from pixels_gym import PixelsGym

def imsquash(state):
    state = state.mean(axis=2)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = skimage.measure.block_reduce(state, (2, 2), np.max)
    state = state.flatten()
    return state

if __name__ == '__main__':
    FRAME_STACK_SIZE = 10
    STATE_SHAPE = 4
    agent = Agent(state_shape=(STATE_SHAPE * FRAME_STACK_SIZE,), num_actions=2,)
    env = PixelsGym(
        env=gym.make('CartPole-v1').unwrapped, 
        frame_stack_size=FRAME_STACK_SIZE, 
        state_formatter=None,
    )

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()

        score, frame = 0, 1
        while not done:
            env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}").format(
            num_samples, episode, high_score, score, frame))

        episode += 1