import copy
import numpy as np
import torch
import torch.nn.functional as F
from dqn import DQN



class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3,
                 epsilon=1.0, decay_rate=0.995, min_eps=0.01,
                 target_update_freq=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_eps = min_eps
        self.action_dim = action_dim
        self.target_update_freq = target_update_freq
        self._train_steps = 0
        
        self.q_net = DQN(state_size=state_dim, action_size=action_dim)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.__init_logger()

    def __init_logger(self):
        self.rewards = []

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_eps)

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_action(self, state):
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values, dim=1).item()
     

    def update(self, batch):
        # `batch` is (states, actions, rewards, next_states, dones), each a numpy
        # array of length batch_size, as produced by the replay buffer.
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)


        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # with torch.no_grad():
        #     next_q_max = self.target_net(next_states).max(dim=1).values
        #     target_q = rewards + self.gamma * next_q_max * (1 - dones)

        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q_max = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q_max * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
   

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.sync_target()
        return loss.item()

