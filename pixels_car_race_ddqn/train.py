import math

import numpy as np
import gym       
import skimage.measure
from skimage import transform
from PIL import Image
from matplotlib import pyplot as plt

from agent import Agent
from ring_gym import RingGym

def imsquash(state):
    # state = state.mean(axis=0)    #   removing color
    # state = state[30:-20, 30:-30] #   cropping
    state = state/255.0
    # state = transform.resize(state, [64,64])

    state = state.transpose((2, 0, 1))
    states = [skimage.measure.block_reduce(state[channel], (2, 2), np.max) for channel in range(3)]
    state = np.stack(states)
    # state = skimage.measure.block_reduce(state, (2, 2), np.max)
    # state = state.flatten()
    return state

def convert_actions(action):
    gas, brake, steer = 0.0, 0.0, 0.0
    if action == 0:     #   forward
        gas = 1.0
    elif action == 1:   #   left
        steer = 1.0
    elif action == 2:   #   right
        steer = -1.0
    elif action == 3:   #   break
        brake = 0.2
    elif action == 3:   #   noop
        gas, brake, steer = 0.0, 0.0, 0.0

    env_actions = np.zeros(3, dtype=np.float)
    env_actions[0] = steer
    env_actions[1] = gas
    env_actions[2] = brake
    
    env_actions.clip(-1.0, 1.0)

    return env_actions

if __name__ == '__main__':
    FRAME_STACK_SIZE = 4
    # STATE_SHAPE_FOR_FC = (4 * FRAME_STACK_SIZE,)
    STATE_SHAPE = (4 * 3, 48, 48)
    agent = Agent(state_shape=STATE_SHAPE, num_actions=5,)
    env = RingGym(
        env=gym.make('CarRacing-v0'),#.unwrapped, 
        frame_stack_size=FRAME_STACK_SIZE, 
        frame_skip_size=2,
        state_formatter=imsquash,
        flatten_states=False,
        rgb_data=True,
    )

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()
        # img = Image.fromarray((state[0]*255).clip(0, 255)).show()
        # quit()

        score, frame = 0, 1
        while not done:
            env.render()

            action = agent.choose_action(state)
            state_, reward, done, info = env.step(convert_actions(action))
            agent.store_memory(state, action, reward, state_, done)
            agent.learn()

            state = state_

            num_samples += 1
            score += reward
            frame += 1

        high_score = max(high_score, score)

        print(("total samples: {}, ep {}: high-score {:12.3f}, score {:12.3f}, epsilon {:12.3f}").format(
            num_samples, episode, high_score, score, agent.epsilon.value()))

        episode += 1