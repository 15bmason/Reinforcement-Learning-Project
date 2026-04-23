import numpy as np
import gymnasium as gym
import seaborn as sns 
import matplotlib.pyplot as plt

# Use a deterministic map (is_slippery=False)
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

# Hyperparameters
episodes = 2000        # Enough time to learn
learning_rate = 0.8    # Learn fast
discount = 0.95        # Care about future rewards
epsilon = 1.0          # Start with 100% exploration
decay_rate = 0.001     # Decay slowly

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

print("Training started...")
goals_reached = 0

for i in range(episodes):
    state, _ = env.reset()
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Exploration (Random) vs Exploitation (Use Q-table)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take action
        new_state, reward, terminated, truncated, _ = env.step(action)

        # If we fell in a hole (terminated with 0 reward), give a penalty (-1)
        # If we hit the goal (reward 1), keep it as 1
        if terminated and reward == 0:
            reward = -1 
        
        # Count successes for your sanity
        if reward == 1:
            goals_reached += 1

        # Update Q-Table (Bellman Equation)
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount * np.max(Q[new_state, :]) - Q[state, action]
        )

        state = new_state

    # Decay epsilon
    epsilon = max(epsilon - decay_rate, 0)

print(f"\nTraining finished. The agent reached the goal {goals_reached} times.")
print("\nFinal Q-Table (Rounded):")
print(np.round(Q, 3))

# VISUALIZE THE STRATEGY
print("\nLearned Policy (0:Left, 1:Down, 2:Right, 3:Up):")
policy = np.argmax(Q, axis=1).reshape((4,4))
print(policy)

# Create a heatmap of the Maximum Q-Value at each state
# This shows the "Value Function" V(s) = max_a Q(s,a)
V_s = np.max(Q, axis=1).reshape((4, 4))

plt.figure(figsize=(6, 5))
sns.heatmap(V_s, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Value V(s)'})
plt.title("Learned Value Function V(s)\n(Red = Goal, Blue = Avoid)")
plt.show()