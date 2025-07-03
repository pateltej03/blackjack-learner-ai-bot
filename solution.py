import gymnasium as gym  # type: ignore
import numpy as np  # type: ignore
import math

env = gym.make("Blackjack-v1", natural=False, sab=False)
# env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode="human")

alpha = 0.3           # Learning rate
gamma = 0.9           # Discount factor
epsilon = 1.0         # Initial exploration rate
min_epsilon = 0.05   
decay_rate = 0.01       
num_episodes = 50000  

q_table = {}
for player_sum in range(4, 22):  
    for dealer_card in range(1, 11):  
        for usable_ace in [0, 1]:  
            for action in range(env.action_space.n): 
                q_table[((player_sum, dealer_card, usable_ace), action)] = 0.0

def get_q_value(state, action):
    
    return q_table.get((state, action), 0.0)

def update_q_table(state, action, reward, next_state):
    max_next_q = max(get_q_value(next_state, a) for a in range(env.action_space.n))
    current_q = get_q_value(state, action)
    q_table[(state, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)

wins = 0
losses = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax([get_q_value(state, a) for a in range(env.action_space.n)]) 

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            if reward > 0:
                reward = 20  
                wins += 1
            elif reward == 0:
                reward = -10  
                losses += 1

        update_q_table(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    if total_reward > 0:
        print(f"Episode {episode + 1}: Win")
    else:
        print(f"Episode {episode + 1}: Loss")

    epsilon = max(min_epsilon, epsilon * math.exp(-decay_rate))

win_rate = wins / num_episodes
print(f"\nTotal Win Rate: {win_rate * 100:.2f}%")

env.close()