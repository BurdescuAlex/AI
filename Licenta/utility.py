import collections
import random


# A simple deque, uniform priority replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

        # Not so optimal for sampling but works...
        self.replayMem = collections.deque(maxlen=buffer_size)

    # Add a new experience to the buffer
    def add(self, state, action, reward, next_state, done):
        self.replayMem.append((state, action, reward, next_state, done))

    # Get a sample batch from the memory
    def sample(self, batch_size):
        if batch_size <= len(self.replayMem):
            return random.sample(self.replayMem, batch_size)
        else:
            assert False

    def __len__(self):
        return len(self.replayMem)
