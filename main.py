# main.py
import argparse
from train import main as train_main
from test import main as test_main
import gym
import yaml
import torch
import os
import time
from agents.sac_agent import SACAgent
import cv2
from gym.wrappers import RecordVideo

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def render_agent():
    # Load config
    with open("config/sac_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Create videos folder (bash)
    os.makedirs("results/videos/", exist_ok=True) # Only if you don't have videos folder at results/, it makes.

    # Set environment for video recording
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder="results/videos/", episode_trigger=lambda x: True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained agent
    agent = SACAgent(
        state_dim, action_dim, action_bound, device,
        lr=config['lr'], gamma=config['gamma'], tau=config['tau'], alpha=config['alpha']
    )
    agent.load("results/saved_model.pth")


    # Run one episode and record
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    rewards = []

    while not done:
        action = agent.select_action(state, eval_mode=True)  # Greedy action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        rewards.append(reward)
        step += 1

    print(f"Video saved: reward = {total_reward:.2f}, steps = {step}")
    time.sleep(1)  # Allow time for video recorder to finalize
    env.close()

    # Convert the saved mp4 to gif with dynamic step overlay
    make_gif_with_dynamic_text(
        "results/videos/rl-video-episode-0.mp4",
        "results/videos/annotated_episode_dynamic.gif",
        rewards
    )

def make_gif_with_dynamic_text(input_video, output_gif, rewards):
    video = VideoFileClip(input_video)

    def add_text(get_frame, t):
        frame = get_frame(t)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        step = int(video.fps * t)
        reward_val = rewards[step] if step < len(rewards) else 0
        text = f"Step: {step}  Reward: {reward_val:.2f}"
        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    video_with_text = video.fl(add_text)
    video_with_text.write_gif(output_gif, fps=15)
    print(f"Annotated GIF saved: {output_gif}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'render'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main()
    elif args.mode == 'test':
        test_main()
    elif args.mode == 'render':
        render_agent()
