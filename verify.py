import csv
from datetime import datetime
import gymnasium as gym
import torch
import numpy as np
from agents import DQNAgent

env = gym.make("LunarLander-v3", render_mode="rgb_array")
# obs, info = env.reset(seed=42)
model_dir="/home/sumit/Documents/Pragya/Lunar Lander/CS-175-Lunar-Lander/model_128_1e4_100k.pth" # we are using this trained model
video_save_path="/home/sumit/Documents/Pragya/Lunar Lander/CS-175-Lunar-Lander/"+model_dir.split("/")[-1][:-4]
env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_save_path,
    name_prefix="eval",
    episode_trigger=lambda x: True
)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
state_dict = torch.load(model_dir)
agent.q_net.load_state_dict(state_dict)
agent.target_net.load_state_dict(state_dict)
agent.epsilon = 0
agent.q_net.eval()
agent.target_net.eval()
reward_col = []
for episode in range(1,101):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(1,1000 + 1):
        action = agent.get_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    reward_col.append(total_reward)
    env.close()

with open(video_save_path+"/verify_rewards_{datetime.now():%Y%m%d_%H%M%S}.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["episode", "return"])
    writer.writerows(enumerate(reward_col, start=1))

print("end")
