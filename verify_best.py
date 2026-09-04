from agents import DQNAgent
from train import evaluate_agent, ENV_NAME, print_summary
import torch
import gymnasium as gym


def main():
    agent = torch.load('model/99-success-model.pth', weights_only=False)
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.q_net.load_state_dict(torch.load("model/99-success-model.pth"))
    agent.target_net.load_state_dict(torch.load("model/99-success-model.pth"))
    agent.q_net.eval()


    eval_results = evaluate_agent(env, agent, 100, 1000, 42)
    print_summary("Evaluation", eval_results)


if __name__ == "__main__":
    main()