from collections import deque

import numpy as np
import gym

'''TODO:
    expose observation space
    expose state space
    expose all gym functions via meta python magic

    allow from frame skipping.
'''

class RingGym:
    ''' tired of making your own implementation of ring buffers 
        and from-pixels framestacking wrappers for open-aigym?
        Be tired no longer. 
        
        Provide the RingGym wrapper class with a frame squasher function and 
        it will automagically give you stacks of frames to work with.
        By choosing RingGym you made life easy for yourself. Good job.'''
    def __init__(self, env, frame_stack_size, state_formatter=None):
        self.frame_stack_size = frame_stack_size
        self.env = env
        self.state_buffer = deque(maxlen=frame_stack_size)

        if state_formatter is None:
            def identity(state):
                return state
            self.state_formatter = identity
        else:
            self.state_formatter = state_formatter

    def _stack_frames(self):
        stack = np.concatenate(list(self.state_buffer))
        return stack

    def step(self, action):
        state_, reward, done, info = self.env.step(action)
        state_ = self.state_formatter(state_)
        self.state_buffer.append(state_)
        stack = self._stack_frames()
        
        return stack, reward, done, info

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        state = self.state_formatter(state)
        for _ in range(self.frame_stack_size):
            self.state_buffer.append(state)
        stack = self._stack_frames()

        return stack
