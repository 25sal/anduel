import pandas as pd
import matplotlib.pyplot as plt

# Filenames of the CSV logs
episode_csv_file = "training_results.csv"
step_csv_file = "step_logs.csv"

# Load the CSV data
episode_data = pd.read_csv(episode_csv_file)
step_data = pd.read_csv(step_csv_file)

# Plotting Total Reward Over Episodes
def plot_training_history(episode_data, save_path="training_history.png"):
    episode_data["Rolling Reward"] = episode_data["Total Reward"].rolling(window=10).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(episode_data["Episode"], episode_data["Total Reward"], label="Total Reward", marker='o', alpha=0.5)
    plt.plot(episode_data["Episode"], episode_data["Rolling Reward"], label="Rolling Avg (10 episodes)", color="red")
    plt.title("Training History: Total Reward Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Training history plot saved at: {save_path}")
    plt.show()

# Plotting the Number of Targets Reached Over Time
def plot_target_reached(episode_data, save_path="target_reached_history.png"):
    episode_data["Target Reached"] = episode_data["Termination Reason"] == "target_reached"
    target_reached_cumulative = episode_data["Target Reached"].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(episode_data["Episode"], target_reached_cumulative, label="Cumulative Targets Reached", color="green", marker='o', alpha=0.7)
    plt.title("Cumulative Targets Reached Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Targets Reached")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Target reached history plot saved at: {save_path}")
    plt.show()

# Plotting Termination Reasons
def plot_termination_reasons(episode_data, save_path="termination_reasons.png"):
    termination_counts = episode_data["Termination Reason"].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(termination_counts.index, termination_counts.values, color=["blue", "orange", "red"], alpha=0.7)
    plt.title("Frequency of Termination Reasons")
    plt.xlabel("Termination Reason")
    plt.ylabel("Count")
    plt.grid(axis='y')
    plt.savefig(save_path)
    print(f"Termination reasons plot saved at: {save_path}")
    plt.show()

# Plotting Steps to Target
def plot_steps_to_target(episode_data, step_data, save_path="steps_to_target.png"):
    # Filter for episodes where the target was reached
    target_reached_episodes = episode_data[episode_data["Termination Reason"] == "target_reached"]["Episode"]
    
    # Filter step_data for steps corresponding to target reached episodes
    steps_to_target = step_data[step_data["Episode"].isin(target_reached_episodes)]["Step"]

    plt.figure(figsize=(10, 6))
    plt.hist(steps_to_target, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Steps to Target")
    plt.ylabel("Frequency")
    plt.title("Distribution of Steps to Reach Target")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Steps to target histogram saved at: {save_path}")
    plt.show()

# Call the functions
plot_training_history(episode_data)
plot_target_reached(episode_data)
plot_termination_reasons(episode_data)
plot_steps_to_target(episode_data, step_data)
