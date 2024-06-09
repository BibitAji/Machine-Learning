import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Actor-Critic Agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.01, gamma=0.99):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)

        # Critic update
        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (1 - done) * self.gamma * next_value
        critic_loss = nn.MSELoss()(value, target.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        advantage = target - value
        log_prob = torch.log(self.actor(state)[action])
        actor_loss = -log_prob * advantage.detach()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Example usage
if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    agent = ActorCriticAgent(state_dim, action_dim)

    # Example interaction with the environment
    for episode in range(100):
        state = np.random.rand(state_dim)
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state = np.random.rand(state_dim)
            reward = np.random.rand()
            done = np.random.rand() < 0.1
            agent.learn(state, action, reward, next_state, done)
            state = next_state
