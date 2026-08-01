import gymnasium as gym
import numpy as np

class CustomLunarLande(gym.Wrapper):

    def __init__(self):
        env = gym.make("LunarLander-v3")
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(6)

    def step(self, action):
        if action <= 3:
            return self.env.step(action)
        
        elif action == 4:
            obs, r1, term, trunc, info = self.env.step(2)
            obs, r2, term, trunc, info = self.env.step(1) 
            return obs, r1 + r2, term, trunc, info

        elif action == 5:
            obs, r1, term, trunc, info = self.env.step(2)
            obs, r2, term, trunc, info = self.env.step(3)
            return obs, r1 + r2, term, trunc, info