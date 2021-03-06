import math

import numpy as np
import gym       
import skimage.measure
from skimage import transform
from PIL import Image


from agent import Agent
from ring_gym import RingGym

def imsquash(state):
    state = state.transpose((2, 0, 1))
    state = state.mean(axis=0)
    state = state[30:-20, 30:-30]
    state = state/255.0
    state = transform.resize(state, [64,64])

    # state = skimage.measure.block_reduce(state, (2, 2), np.max)
    # state = skimage.measure.block_reduce(state, (2, 2), np.max)
    # state = state.flatten()
    return state

if __name__ == '__main__':
    FRAME_STACK_SIZE = 4
    STATE_SHAPE_FOR_FC = (4 * FRAME_STACK_SIZE,)
    STATE_SHAPE = (4, 64, 64)
    agent = Agent(state_shape=STATE_SHAPE, num_actions=2,)
    env = RingGym(
        env=gym.make('CartPole-v1').unwrapped, 
        frame_stack_size=FRAME_STACK_SIZE, 
        frame_skip_size=2,
        state_formatter=imsquash,
        flatten_states=False,
    )

    high_score = -math.inf
    episode = 0
    num_samples = 0
    while True:
        done = False
        state = env.reset()
        # img = Image.fromarray((state[0]*255).clip(0, 255)).show()

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