import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class PlotCallback(BaseCallback):
    def __init__(self, window: int = 1, save_path: str = "training_rewards.png",
                 title: str = "Training Rewards", verbose=0):
        super().__init__(verbose)
        self.window     = window
        self.save_path  = save_path
        self.title      = title
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
        return True

    def _on_training_end(self) -> None:
        data = np.array(self.episode_rewards)
        total_reward = np.sum(data)
        reward_std = np.std(data)

        if self.window > 1 and len(data) >= self.window:
            ma = np.convolve(data, np.ones(self.window) / self.window, mode="valid")
        else:
            ma = data

        # Plotting
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ma)
        ax.set_title(self.title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)
        fig.savefig(self.save_path)
        plt.close(fig)

        # Print reward summary
        print(f"✅ Training curve saved to: {self.save_path}")
        print(f"📊 Total reward: {total_reward:.2f}")
        print(f"📉 Reward variation (std dev): {reward_std:.2f}")
