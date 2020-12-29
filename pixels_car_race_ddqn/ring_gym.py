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
    ''' Are you tired of rolling your own ring buffers and framestacking wrappers for open-aigym?
        Be tired no longer my friend. Introducing, the NEW RingGym wrapper class. 
        
        Just provide a state formatter function in case you want to monochrome or shrink your input images, 
        and presto, the RingGym wrapper class will automagically give you stacks of frames to work with.

        By choosing RingGym you made life easy for yourself. Good job.'''
    def __init__(self, env, frame_stack_size, 
            frame_skip_size=1, state_formatter=None, flatten_states=False, rgb_data=False):
        '''-set flatten_states=True if you are using non convolutional data
        '''
        self.frame_stack_size = frame_stack_size
        self.env = env
        self.frame_skip_size = frame_skip_size
        self.buffer_size = self.frame_stack_size * (frame_skip_size + 1)
        self.state_buffer = deque(maxlen=self.buffer_size)
        self.flatten_states = flatten_states
        self.rgb_data = rgb_data

        if state_formatter is None:
            def identity(state):
                return state
            self.state_formatter = identity
        else:
            self.state_formatter = state_formatter

    def _stack_frames(self):
        # stack = np.concatenate(list(self.state_buffer))
        states = list(self.state_buffer)[::-(self.frame_skip_size)]
        states = states[:self.frame_stack_size]
        stack = np.stack(states)
        if self.rgb_data:
            stack = stack.reshape((self.frame_stack_size * 3, *stack.shape[-2:]))
        if self.flatten_states:
            stack = stack.flatten()
        
        return stack

    def step(self, action):
        state_, reward, done, info = self.env.step(action)

        #   this is the custom cpole version. ideally we wouldnt have to do this.
        # state_ = self.env.render(mode='rgb_array')

        state_ = self.state_formatter(state_)
        self.state_buffer.append(state_)
        stack = self._stack_frames()
        
        return stack, reward, done, info

    def render(self):
        self.env.render()

    def reset(self):
        state = self.env.reset()
        
        #   this is the custom cpole version. ideally we wouldnt have to do this.
        # state = self.env.render(mode='rgb_array')

        state = self.state_formatter(state)
        for _ in range(self.buffer_size):
            self.state_buffer.append(state)
        stack = self._stack_frames()

        return stack
