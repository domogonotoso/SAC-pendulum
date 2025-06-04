# test.py
import gym
import yaml
import numpy as np
from agents.sac_agent import SACAgent
import torch

def main():
    # Load hyperparameters from config file
    with open('config/sac_config.yaml') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize SAC agent
    agent = SACAgent(
        state_dim, action_dim, action_bound, device,
        lr=config['lr'], gamma=config['gamma'], tau=config['tau'], alpha=config['alpha']
    )

    # Load the trained model
    agent.load("results/saved_model.pth")  # 🔥 Make sure this path matches your saved model

    num_test_episodes = 10
    rewards = []

    for episode in range(1, num_test_episodes + 1):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            # Select action without exploration noise (evaluation mode)
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step += 1

        rewards.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {step}")

    env.close()
    print(f"\nAverage Reward over {num_test_episodes} episodes: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    main()
