import csv
from datetime import datetime
import gymnasium as gym

def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_save_path = "videos"
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_save_path,
        name_prefix="baseline",
        episode_trigger=lambda x: x < 5  # only record first 5 episodes, adjust as needed
    )

    reward_col = []
    for episode in range(1, 101):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1, 1000 + 1):
            action = env.action_space.sample()
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