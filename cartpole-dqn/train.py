import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from model import DQN

# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ===== Training Step =====
def train_step():
    if len(memory) < batch_size:
        return

    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = target_net(next_states).max(1)[0].detach()

    target_q = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ===== Environment Setup =====
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = ReplayBuffer()

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 128
target_update = 10
episodes = 400

# ===== Training Loop =====
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = policy_net(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode+1}, Score: {total_reward}, Epsilon: {epsilon:.3f}")

env.close()


torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
print("Model saved!")