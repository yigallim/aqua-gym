import numpy as np
import random
from collections import defaultdict
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from envs.aquaculture_env import AquacultureEnv

class DiscretizedDynaQAgent:
    def __init__(
        self,
        env,
        alpha=1e-3,
        gamma=0.99,
        planning_steps=10,
        obs_bins=10,
        buffer_size=10000,
        batch_size=32,
        replay_freq=5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.2,
        total_timesteps=300 * 180
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.planning_steps = planning_steps
        self.obs_bins = obs_bins
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_freq = replay_freq
        self.experience_buffer = deque(maxlen=buffer_size)
        self.episode_rewards = []

        # Exploration
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.epsilon = exploration_initial_eps
        self.global_step = 0

        # Discretize action space
        self.feed_bins = 40
        self.temp_bins = 16
        self.air_bins = 10
        self.action_space = [
            (feed, temp, air)
            for feed in np.linspace(env.action_space.low[0], env.action_space.high[0], self.feed_bins)
            for temp in np.linspace(env.action_space.low[1], env.action_space.high[1], self.temp_bins)
            for air in np.linspace(env.action_space.low[2], env.action_space.high[2], self.air_bins)
        ]

        self.q_table = defaultdict(lambda: np.zeros(len(self.action_space)))
        self.model = {}

        self.obs_space_low = env.observation_space.low
        self.obs_space_high = env.observation_space.high
        self.obs_space_bins = [
            np.linspace(self.obs_space_low[i], self.obs_space_high[i], obs_bins)
            for i in range(env.observation_space.shape[0])
        ]

    def discretize_obs(self, obs):
        return tuple(int(np.digitize(obs[i], self.obs_space_bins[i])) for i in range(len(obs)))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.action_space) - 1)
        return np.argmax(self.q_table[state])

    def update_q(self, state, action_idx, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state][action_idx] += self.alpha * (td_target - self.q_table[state][action_idx])

    def learn_model(self, state, action_idx, reward, next_state):
        self.model[(state, action_idx)] = (reward, next_state)

    def planning(self):
        for _ in range(self.planning_steps):
            if not self.model:
                return
            (s, a), (r, s_next) = random.choice(list(self.model.items()))
            self.update_q(s, a, r, s_next)

    def sample_and_update(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.experience_buffer, self.batch_size)
        for (s, a, r, s_next) in minibatch:
            self.update_q(s, a, r, s_next)

    def train(self, episodes=300, plot=True, verbose=True):
        rewards = []

        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.discretize_obs(obs)
            total_reward = 0
            done = False
            truncated = False

            while not (done or truncated):
                action_idx = self.choose_action(state)
                action = np.array(self.action_space[action_idx], dtype=np.float32)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                next_state = self.discretize_obs(next_obs)

                self.update_q(state, action_idx, reward, next_state)
                self.learn_model(state, action_idx, reward, next_state)
                self.experience_buffer.append((state, action_idx, reward, next_state))
                if self.global_step % self.replay_freq == 0:
                    self.sample_and_update()
                self.planning()

                state = next_state
                total_reward += reward
                self.global_step += 1

            rewards.append(total_reward)
            self.episode_rewards.append(total_reward)

            # Decay epsilon
            progress = min(1.0, ep / (episodes * self.exploration_fraction))
            self.epsilon = self.exploration_initial_eps - progress * (self.exploration_initial_eps - self.exploration_final_eps)

            if verbose:
                print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.4f}")

        if verbose:
            print(f"\nTotal Cumulative Reward after {episodes} Episodes: {sum(rewards):.2f}")
            print(f"Average Reward per Episode: {np.mean(rewards):.2f}")
            print(f"Reward Variation (Std Dev): {np.std(rewards):.2f}")

        if plot:
            self.plot_rewards(rewards)

        return rewards

    def plot_rewards(self, rewards):
        region_name = getattr(self.env, "region", "unknown").capitalize()
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label=f"Region: {region_name}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Dyna-Q Performance in {region_name} Region | Final ε = {self.exploration_final_eps:.3f}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
