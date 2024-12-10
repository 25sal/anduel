import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import os

KMH_TO_NMPS = 1.852 / 3600  # Convert km/h to nautical miles per second

# Define the ATC Environment
class ATCEnv(gym.Env):
    def __init__(self):
        super(ATCEnv, self).__init__()
        self.grid_size = 40
        self.no_fly_zone_center = np.array([20, 20])
        self.no_fly_zone_radius = 5
        self.separation_threshold = 2.5
        self.max_steps = 700
        self.aircraft_start = np.array([20, 40])
        self.aircraft_target = np.array([20, 0])
        self.initial_speed_kmh = 300
        self.initial_angle_deg = random.uniform(0, 360)  # Random angle initialization

        self.action_space = gym.spaces.Discrete(3)  # [-10, 0, +10] for angle adjustment
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # Distance to target replaces speed
            high=np.array([self.grid_size, self.grid_size, self.grid_size, 360, self.grid_size, self.grid_size]),
            dtype=np.float32
        )

    def reset(self):
        self.aircraft_position = self.aircraft_start.copy()
        self.previous_distance_to_target = self._calculate_distance_to_target()
        self.current_step = 0
        self.angle = random.uniform(0, 360)  # Reinitialize with random angle
        return self._get_obs()

    def _get_obs(self):
        distance_to_target = self._calculate_distance_to_target()
        return np.hstack([
            self.aircraft_position,
            distance_to_target,
            self.angle,
            self.no_fly_zone_center
        ])

    def _calculate_distance_to_target(self):
        return np.linalg.norm(self.aircraft_position - self.aircraft_target)

    def step(self, action):
        self.current_step += 1

        # Adjust the angle based on action
        angle_adjustment = (action - 1) * 5
        self.angle = (self.angle + angle_adjustment) % 360

        # Update position
        speed_nmps = self.initial_speed_kmh * KMH_TO_NMPS
        angle_rad = np.deg2rad(self.angle)
        self.aircraft_position[0] += speed_nmps * np.cos(angle_rad)
        self.aircraft_position[1] += speed_nmps * np.sin(angle_rad)

        # Clip position to grid boundaries
        self.aircraft_position = np.clip(self.aircraft_position, 0, self.grid_size)

        # Calculate reward and check termination
        total_reward, rewards_breakdown, done, truncate, termination_reason = self._calculate_reward()

        return self._get_obs(), total_reward, done, truncate, {
            "termination_reason": termination_reason,
            "position": self.aircraft_position,
            "angle": self.angle,
            **rewards_breakdown
        }

    def _calculate_reward(self):
        rewards_breakdown = {
            "move_reward": 0,
            "no_fly_zone_penalty": 0,
            "boundary_penalty": 0,
            "target_reward": 0
        }
        done = False
        truncate = False
        termination_reason = ""

        # Distance to target
        distance_to_target = np.linalg.norm(self.aircraft_position - self.aircraft_target)

        # Reward for moving closer to the target
        if distance_to_target < self.previous_distance_to_target:
            rewards_breakdown["move_reward"] = 5
        else:
            rewards_breakdown["move_reward"] = -2

        # Penalty for entering the no-fly zone
        distance_to_no_fly_zone = np.linalg.norm(self.aircraft_position - self.no_fly_zone_center)
        if distance_to_no_fly_zone < self.no_fly_zone_radius + self.separation_threshold:
            rewards_breakdown["no_fly_zone_penalty"] = -200
            done = True
            termination_reason = "no_fly_zone"

        # Penalty for hitting grid boundaries
        if self.aircraft_position[0] == 0 or self.aircraft_position[0] == self.grid_size:
            rewards_breakdown["boundary_penalty"] = -50
            truncate = True
            termination_reason = "boundary"

        # Reward for reaching the target
        if self.aircraft_position[1] == 0:
            rewards_breakdown["target_reward"] = 200
            done = True
            termination_reason = "target_reached"

        # Max steps termination
        if self.current_step >= self.max_steps and not done:
            truncate = True
            termination_reason = "max_steps"

        # Total reward
        total_reward = sum(rewards_breakdown.values())
        self.previous_distance_to_target = distance_to_target

        return total_reward, rewards_breakdown, done, truncate, termination_reason


# DQN Class
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Saliency Map Function
def compute_saliency_map(model, state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).requires_grad_(True)
    q_values = model(state_tensor)
    q_values.max().backward()
    saliency = state_tensor.grad.abs().squeeze().detach().numpy()
    saliency /= saliency.max() + 1e-8  # Normalize to [0, 1]
    return saliency


# Plotting Functions
def plot_trajectory(step_logs, episode, save_path="trajectories"):
    os.makedirs(save_path, exist_ok=True)
    trajectory = np.array([[log["position"][0], log["position"][1]] for log in step_logs])
    plt.figure(figsize=(8, 8))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label='Path')
    plt.scatter(20, 40, color='blue', label='Start')
    plt.scatter(20, 0, color='green', label='Target')
    plt.gca().add_patch(plt.Circle((20, 20), 7.5, color='red', alpha=0.3, label='No-Fly Zone'))
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.legend()
    plt.title(f"Trajectory - Episode {episode}")
    plt.savefig(f"{save_path}/trajectory_{episode}.png")
    plt.close()


def plot_reward_decomposition(reward_components, episode, save_path="reward_decompositions"):
    os.makedirs(save_path, exist_ok=True)
    labels = reward_components.keys()
    values = reward_components.values()
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, alpha=0.7, color=['red', 'orange', 'green'])
    plt.title(f"Reward Decomposition - Episode {episode}")
    plt.xlabel("Reward Component")
    plt.ylabel("Total Reward Contribution")
    plt.grid(True)
    plt.savefig(f"{save_path}/reward_decomposition_{episode}.png")
    plt.close()


def plot_saliency_map(saliency, episode, step, feature_names, save_path="saliency_maps"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(saliency)), saliency, color='blue', alpha=0.7)
    plt.xticks(range(len(saliency)), feature_names, rotation=45)
    plt.title(f"Saliency Map - Episode {episode}, Step {step}")
    plt.xlabel("Feature")
    plt.ylabel("Saliency")
    plt.grid(True)
    plt.savefig(f"{save_path}/saliency_episode_{episode}_step_{step}.png")
    plt.close()


# Training Loop
env = ATCEnv()
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

q_network = DQN(input_size, output_size)
target_network = DQN(input_size, output_size)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=0.001)
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 20000
replay_buffer = []
batch_size = 64
max_buffer_size = 10000

feature_names = ["Aircraft X", "Aircraft Y", "Distance to Target", "Angle", "NFZ X", "NFZ Y"]

episode_csv_file = "training_results.csv"
step_csv_file = "step_logs.csv"

# Logging Setup
with open(episode_csv_file, mode="w", newline="") as episode_file, open(step_csv_file, mode="w", newline="") as step_file:
    episode_writer = csv.writer(episode_file)
    step_writer = csv.writer(step_file)
    episode_writer.writerow(["Episode", "Total Reward", "Steps", "Termination Reason"])
    step_writer.writerow(["Episode", "Step", "X", "Y", "Angle", "Move Reward", "No-Fly Zone Penalty", "Boundary Penalty", "Target Reward"])

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        truncate = False
        termination_reason = ""
        step_logs = []
        reward_components = {"safety": 0, "efficiency": 0, "goal_achievement": 0}

        while not done and not truncate:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done, truncate, step_info = env.step(action)
            steps += 1
            episode_reward += reward
            decomposed = {
                "safety": step_info["no_fly_zone_penalty"] + step_info["boundary_penalty"],
                "efficiency": step_info["move_reward"],
                "goal_achievement": step_info["target_reward"]
            }
            for key in reward_components:
                reward_components[key] += decomposed[key]

            # Log step data
            step_logs.append({
                "position": step_info["position"],
                "angle": step_info["angle"],
                **step_info
            })

            # Saliency Map
            saliency = compute_saliency_map(q_network, state)
            plot_saliency_map(saliency, episode + 1, steps, feature_names)

            state = next_state

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones)

                q_values = q_network(states)
                next_q_values = target_network(next_states)

                q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_q_values = next_q_values.max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = (q_values - target_q_values.detach()).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Write Episode Data
        episode_writer.writerow([episode + 1, episode_reward, steps, step_info["termination_reason"]])

        # Write Step Logs
        for step, log in enumerate(step_logs, start=1):
            step_writer.writerow([episode + 1, step, log["position"][0], log["position"][1], log["angle"],
                                  log["move_reward"], log["no_fly_zone_penalty"],
                                  log["boundary_penalty"], log["target_reward"]])

        # Plot Trajectory and Reward Decomposition
        plot_trajectory(step_logs, episode + 1)
        plot_reward_decomposition(reward_components, episode + 1)

        print(f"Episode {episode + 1}, Total Reward: {episode_reward}, Steps: {steps}, Termination: {step_info['termination_reason']}")
