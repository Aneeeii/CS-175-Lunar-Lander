import numpy as np

class ReplayBuffer:

    STATE_DIM = 8  # state is an 8-dimensional vector as seen in the documentation

    def __init__(self, cap=200000):
        # uses circular buffer array for storage for efficiency
        # size tracks the valid data slots for sampling before the buffer is full
        self.state = np.zeros((cap, self.STATE_DIM))
        self.action = np.zeros((cap,))
        self.reward = np.zeros((cap,))
        self.next_state = np.zeros((cap, self.STATE_DIM))
        self.done = np.zeros((cap,))
        self.capacity = cap
        self.head = 0
        self.size = 0

    def store_transition(self, state, action, reward, next_state, done):
        # store new transition
        self.state[self.head] = state
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.next_state[self.head] = next_state
        self.done[self.head] = done

        # update size if applicable
        if self.size != self.capacity:
            self.size += 1

        # increment head (sets back to 0 once capacity is reached)
        self.head = (self.head + 1) % self.capacity

    def sample(self, n):
        # return value is [[all states], [all actions], ...]
        idxs = np.random.randint(0, self.size, size=n)
        return (
            self.state[idxs],
            self.action[idxs],
            self.reward[idxs],
            self.next_state[idxs],
            self.done[idxs],
        )