# 🌊 AQUA-GYM: Reinforcement Learning for Smart Aquaculture Management

Optimize fish feeding, energy use, and water quality using Reinforcement Learning (RL) to enhance sustainability and profitability in inland tank aquaculture.

<video src="./demo.mp4" controls width="600">
  Your browser does not support the video tag.
</video>

---

## 🧠 Core Concepts

### ✅ Objective

1. Maximize **profitability** via feed optimization.
2. Reduce **feed wastage** and overfeeding.
3. Enhance **water quality** (DO, UIA, Temp).
4. Improve **energy efficiency** in aeration/heating.

### 🎯 Reinforcement Learning Setup

| Component       | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| **State (S)**   | Biomass, Fish Count, Temperature, DO, UIA                              |
| **Action (A)**  | Feeding Rate, Target Temp, Aeration Rate                               |
| **Reward (R)**  | Fish value gain – Feed cost – Energy cost                              |
| **Agent (π)**   | Learns feeding & control policy from real-time feedback                |
| **Environment** | OpenAI Gym-compatible, built using `aquaculture_env.py` and DEB theory |

---

## 🤖 RL Algorithms Used

| Algorithm  | Type                              | Description                                            |
| ---------- | --------------------------------- | ------------------------------------------------------ |
| **TD3**    | Model-Free                        | Best for continuous control like feeding rate and temp |
| **SAC**    | Model-Free                        | Entropy-based for better exploration                   |
| **DQN**    | Model-Free                        | Discrete actions, less efficient for continuous tasks  |
| **Dyna-Q** | Hybrid (Model-Based + Model-Free) | Combines simulation & real data learning               |

---

## 📈 Evaluation Metrics

| Metric                          | Purpose                             |
| ------------------------------- | ----------------------------------- |
| **Total Reward**                | Overall performance indicator       |
| **Feed Conversion Ratio (FCR)** | Feed efficiency (lower = better)    |
| **Specific Growth Rate (SGR)**  | Growth performance                  |
| **Energy Efficiency**           | Value gain per energy cost          |
| **Profit Margin**               | Profitability of the farm operation |

---

## 🌍 Deployment Environments

The RL agents were tested under 3 real-world region profiles:

| Region            | Challenge Level | Differences                 |
| ----------------- | --------------- | --------------------------- |
| Guangdong 🇨🇳      | Hard            | High energy cost            |
| North Sulawesi 🇮🇩 | Medium          | Balanced conditions         |
| Kafr El-Shaikh 🇪🇬 | Easy            | Cheap energy, low feed cost |

---

## 📊 Key Findings

- **SAC** outperformed others across all KPIs (avg. 90.6%)
- **Dyna-Q** performed well in harder scenarios with less training data
- **TD3** provided stable but moderate performance
- **DQN** was less suitable due to discrete action limitations

---

## 📁 Project Structure

```text
aqua-gym/
├── agent/              # Reinforcement learning agents (e.g., Dyna-Q)
├── assets/             # Fish images and tank sound effects
├── envs/               # Aquaculture simulation environments (Gym-style)
├── model/              # Fish growth and environment models
├── plots/              # Hyperparameter tuning & exploration experiments
├── saved_model/        # Trained RL models parameters
├── utils/              # Configs, metric calculations, plotting tools
├── parameters.yaml     # Environment configuration
├── FINAL_EVALUATION.ipynb  # Model comparison notebook
├── test_*.ipynb        # Testing and evaluation notebooks
├── README.md           # Project documentation

```
