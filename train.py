# train.py
import gym
import yaml
import numpy as np
from agents.sac_agent import SACAgent
from utils.replay_buffer import ReplayBuffer
from utils.plot import plot_rewards
import torch
import os

def main():
    # Load config
    with open('config/sac_config.yaml') as f:
        config = yaml.safe_load(f)

    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = SACAgent(
        state_dim, action_dim, action_bound, device,
        lr=config['lr'], gamma=config['gamma'], tau=config['tau'], alpha=config['alpha']
    )

    replay_buffer = ReplayBuffer(capacity=1000000)

    num_episodes = config['num_episodes']
    batch_size = config['batch_size']

    reward_history = []

    # Create results directory if it doesn't exist
    os.makedirs("results/", exist_ok=True)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False    
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.update(replay_buffer, batch_size)

        reward_history.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"10-episode avg: {avg_reward:.2f}")

    # Save model checkpoint
    agent.save("results/saved_model.pth") 

    # Save reward plot
    plot_rewards(reward_history, save_path='results/rewards_plot.png')

    env.close()
    print("Training finished. Model and reward plot saved.")

if __name__ == "__main__":
    main()
