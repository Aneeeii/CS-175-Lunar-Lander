from train import run_baseline
import torch
import gymnasium as gym
from train import ENV_NAME
from agents import DQNAgent
import csv
from datetime import datetime


def main():
    video_save_path = "videos"
    agent = torch.load('model/99-success-model.pth', weights_only=False)
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_save_path,
            name_prefix="best",
            episode_trigger=lambda x: x < 5  # only record first 5 episodes, adjust as needed
        )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.q_net.load_state_dict(torch.load("model/99-success-model.pth"))
    agent.target_net.load_state_dict(torch.load("model/99-success-model.pth"))
    agent.epsilon = 0
    agent.q_net.eval()
    agent.target_net.eval()

    reward_col = []
    for episode in range(1, 101):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1, 1000 + 1):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        reward_col.append(total_reward)

    env.close()

    with open(video_save_path + f"/verify_rewards_{datetime.now():%Y%m%d_%H%M%S}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "return"])
        writer.writerows(enumerate(reward_col, start=1))

    print("end")


if __name__ == "__main__":
    main()