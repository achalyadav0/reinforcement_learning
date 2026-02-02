import gymnasium as gym
import torch
import time
from model import DQN

# Create environment with rendering ON
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size)
model.load_state_dict(torch.load("dqn_cartpole.pth"))
model.eval()

episodes = 5

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward

        time.sleep(0.02)  # Slow down so you can see

    print(f"Episode {episode+1} Score: {total_reward}")

env.close()