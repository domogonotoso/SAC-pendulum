## Controlling Pendulum by SAC
![asdf](results/videos/annotated_episode_dynamic.gif)

## SAC (Soft Actor Critic), What is it?
Soft Actor-Critic (SAC) is off-policy RL algorithm for continuous action spaces.  
It especially uses an actor network to output mean and standard deviation of actions, creating a Gaussian policy. 
And estimate its q_value for knowing how action at the situation is good.
SAC also includes entropy regularization at Loss function to balance exploration and exploitation, making the policy soft. 

## 📁 Project Structure

```text
sac-pendulum/
├── agents/
│   └── sac_agent.py         # SAC agent with actor-critic update logic and replay buffer
├── models/
│   ├── actor.py             # Actor network definition
│   └── critic.py            # Critic (Q-value) network definition
├── utils/
│   └── plot.py              # Training curve plotting utilities
├── config/
│   └── sac_config.yaml      # Hyperparameter configuration file
├── videos/                  # Recorded environment videos
│   └── rl-video-episode-0.mp4
├── results/
│   ├── rewards_plot.png     # Reward curve plot
│   └── saved_model.pth      # Final trained model checkpoint
├── main.py                  # Entry point for training/eval/render
├── train.py                 # Training loop
├── test.py                  # Evaluation script
├── requirements.txt         # Required packages
├── README.md                # Project overview and usage
└── .gitignore               # Comon ignores (pycache, videos, etc.)


```



## ✅ Implemented Features
1. block overestimating q-value
```python
        # sac_agent.py, line 97
        q1_pi = self.critic_1(state, action_sample)
        q2_pi = self.critic_2(state, action_sample)
        min_q_pi = torch.min(q1_pi, q2_pi)
```
2. Soft update
```python
        # sac_agent.py, line 107
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
```
3. video with step and reward per frames
```python
        # main.py, line 72
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
```

4. Use distribution at train mode to explore and use pdf(for entropy) at Loss function
```python
        # sac_agent.py, line 37
        if eval_mode:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
```

5. Get a log_prob from probability density function, adjust scaled by multipling Jacobian, use approximate formular to block overflow 
```python
        # sac_agent.py, line 60, 92
            log_prob = dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - next_action - F.softplus(-2 * next_action))).sum(dim=-1, keepdim=True)
```

6. main.py  train test render

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the agent
python main.py --mode train

# Test the trained agent
python main.py --mode test

# Render and record video
python main.py --mode render
```


## Curious 헷갈렸던거 어려웠던

1. Q. Why it doesn't use expectation of entropy but use the part of it?  
$H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$            log_prob = log π(a|s)
   A. We use MSELoss, and it sum data of a batch and divide. It can be a expectiation of entropy approximately.

2. Comprehending
```python
        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=-1, keepdim=True)
```

We put the value of action_sample to gaussian distribution we used, which is called probability density function. But we multiply Jacobian to it, because action-space is different with originally gaussian distribution's space. 

This is an example of probability density function from action-space and gaussian distribution that has mean = 0 and standard deviation = 1 
![pdf](photo&gif\pdf.png)

For fix this scale, we multiply jacobian and there it is.
$$
\pi(a) = \pi_u(u) \cdot \left|\frac{du}{da}\right|
$$
$$
\log \pi(a) = \log \pi_u(u) + \log \left|\frac{du}{da}\right|
$$
The derivative of the transformation is:

$$
\frac{\partial a}{\partial u} = 1 - \tanh^2(u)   \text{, }  \frac{\partial u}{\partial a} = \frac{1}{1 - \tanh^2(u)}
$$


$$
\log \left|\frac{du}{da}\right| = - \log(1 - \tanh^2(u))
$$



$$
-\log(1 - \tanh^2(u)) = 2 \log \cosh(u) \quad 
$$



$$
\log \cosh(u) = \log(2) - u - \mathrm{softplus}(-2u) (\text{where } \mathrm{softplus}(z)=\log(1+e^z))
$$



$$
-\log(1 - \tanh^2(u)) = 2(\log(2) - u - \mathrm{softplus}(-2u))
$$


So Jacobian applied to python code.
$$
\log \left|\frac{du}{da}\right| = 2(\log(2) - u - \mathrm{softplus}(-2u))
$$


```python
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=-1, keepdim=True)
```
(from 
$H(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s)]$
, we use $-\log \pi(a)$)