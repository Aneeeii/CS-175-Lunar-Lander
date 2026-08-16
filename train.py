import argparse
import csv
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from agents import DQNAgent
from replay_buffer import ReplayBuffer
import utils


ENV_NAME = "LunarLander-v3"
SUCCESS_REWARD = 200


def run_baseline(env, episodes, max_steps, seed):
    results = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(1, max_steps + 1):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        results.append({"phase": "baseline", "episode": episode, "return": total_reward})

    return results


def train_agent(env, episodes, max_steps, seed, batch_size, learning_starts, stop_after=10):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    buffer = ReplayBuffer()
    results = []
    steps_no_improv = 0
    AVG_REWARD_WINDOW = 20
    curr_avg = None
    lthreshold_ep = -1

    start_time = time.perf_counter()

    for episode in range(1, episodes + 1):
        if lthreshold_ep != -1 and episode - lthreshold_ep > AVG_REWARD_WINDOW:
            curr_avg = utils.calculate_avg_rewards(results[-AVG_REWARD_WINDOW:])

        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(1, max_steps + 1):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.store_transition(state, action, reward, next_state, done)

            if buffer.size >= learning_starts:
                if lthreshold_ep == -1:
                    lthreshold_ep = episode
                batch = buffer.sample(batch_size)
                agent.update(batch)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon_decay()
        results.append({"phase": "train", "episode": episode, "return": total_reward})

        # if episode % 100 == 0:
        #     torch.save(agent.q_net.state_dict(), f"checkpoint_ep{episode}.pth")
        #     print(f"Checkpoint saved at episode {episode}")

        torch.save(agent.q_net.state_dict(), "model.pth")

        if episode % 25 == 0:
            print_summary("Training", results)

        if episode % 1000 == 0:
            print(f"Training episode {episode}. Time elapsed = {time.perf_counter() - start_time:0.4f})

        if curr_avg is not None and curr_avg > utils.calculate_avg_rewards(results[-AVG_REWARD_WINDOW:]):
            steps_no_improv += 1
        else:
            steps_no_improv = 0

        if steps_no_improv >= stop_after:
            break

    # torch.save(agent.q_net.state_dict(), "model.pth")
    # print("Model saved to model.pth")

    run_time = time.perf_counter() - start_time
    print(f"Agent ran for {run_time // 60} minutes and {run_time % 60}seconds.")

    return agent, results


def evaluate_agent(env, agent, episodes, max_steps, seed):
    old_epsilon = agent.epsilon
    agent.epsilon = 0
    results = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0

        for step in range(1, max_steps + 1):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        results.append({"phase": "eval", "episode": episode, "return": total_reward})

    agent.epsilon = old_epsilon
    return results


def print_summary(label, results):
    returns = [row["return"] for row in results]
    success_rate = np.mean([score >= SUCCESS_REWARD for score in returns])
    print(
        f"{label}: avg return = {np.mean(returns):.2f}, "
        f"success rate = {success_rate * 100:.1f}%"
    )


def save_log(rows, path):
    path.parent.mkdir(exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["phase", "episode", "return"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "train", "all"], default="all")
    parser.add_argument("--baseline-episodes", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=Path, default=Path("logs/training_log.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = []

    env = gym.make(ENV_NAME)
    env.action_space.seed(args.seed)

    if args.mode in ("baseline", "all"):
        baseline_results = run_baseline(
            env, args.baseline_episodes, args.max_steps, args.seed
        )
        all_results.extend(baseline_results)
        print_summary("Random baseline", baseline_results)

    if args.mode in ("train", "all"):
        agent, train_results = train_agent(
            env,
            args.train_episodes,
            args.max_steps,
            args.seed,
            args.batch_size,
            args.learning_starts,
        )
        all_results.extend(train_results)
        print_summary("Training final", train_results)

        eval_results = evaluate_agent(env, agent, args.eval_episodes, args.max_steps, args.seed)
        all_results.extend(eval_results)
        print_summary("Evaluation", eval_results)

    save_log(all_results, args.log_file)
    env.close()
    print(f"Saved log to {args.log_file}")


if __name__ == "__main__":
    main()
