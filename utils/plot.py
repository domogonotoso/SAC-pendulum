import matplotlib.pyplot as plt

def plot_rewards(rewards, save_path='results/rewards_plot.png'):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Reward Curve')
    plt.savefig(save_path)
    plt.close()
