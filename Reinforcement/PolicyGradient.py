import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.policy_net(state)
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
        return action

    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def train(self):
        R = 0
        rewards = []
        for r in self.episode_rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(0, R)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)

        loss = 0
        for log_prob, reward in zip(self.episode_log_probs, rewards):
            loss -= log_prob * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

# Example usage:
if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    agent = PolicyGradientAgent(state_dim, action_dim)

    # Example interaction with the environment
    for episode in range(100):
        state = np.random.rand(state_dim)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state = np.random.rand(state_dim)
            reward = np.random.rand()
            done = np.random.rand() < 0.1
            agent.store_transition(state, action, reward)
            state = next_state

        agent.train()
