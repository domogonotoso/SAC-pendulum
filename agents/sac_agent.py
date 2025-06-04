import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.actor import Actor
from models.critic import Critic

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, device, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Networks
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)

        # Initialize target networks
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

        self.action_bound = action_bound  # e.g. [-2, 2]

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mean, std = self.actor(state)
        if eval_mode:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
        action = torch.tanh(action) * self.action_bound  # squash action to valid range, which is assigned by environment
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Sample next action (a')
        with torch.no_grad():
            next_mean, next_std = self.actor(next_state)
            dist = torch.distributions.Normal(next_mean, next_std)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            # Correct log_prob for tanh squashing
            log_prob -= (2 * (np.log(2) - next_action - F.softplus(-2 * next_action))).sum(dim=-1, keepdim=True)
            next_action = torch.tanh(next_action) * self.action_bound

            # Target Q
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Critic 1 loss
        current_q1 = self.critic_1(state, action)
        critic_1_loss = F.mse_loss(current_q1, target_q)

        # Critic 2 loss
        current_q2 = self.critic_2(state, action)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor loss
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        # Correct log_prob for tanh squashing
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=-1, keepdim=True)
        action_sample = torch.tanh(action_sample) * self.action_bound

        q1_pi = self.critic_1(state, action_sample)
        q2_pi = self.critic_2(state, action_sample)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])
        print("Model loaded successfully.")

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
        }, path)
        print("Model saved successfully.")

