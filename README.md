프레임별 reward/step/epsilon 표시




```text
sac-pendulum/
├── agents/
│   └── sac_agent.py         # SAC agent with actor-critic update logic and replay buffer
├── models/
│   ├── actor.py             # Actor network definition
│   ├── critic.py            # Critic (Q-value) network definition
│   └── value.py             # (Optional) Value network if using soft value update separately
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
└── .gitignore               # Common ignores (pycache, videos, etc.)


```

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```