import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.n_actions)
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)

# Example usage:
if __name__ == "__main__":
    n_states = 5
    n_actions = 2
    q_learning = QLearning(n_states, n_actions)

    # Example interaction with the environment
    for episode in range(100):
        state = np.random.randint(0, n_states)
        while True:
            action = q_learning.choose_action(state)
            next_state = (state + action) % n_states
            reward = 1 if next_state == n_states - 1 else 0
            q_learning.learn(state, action, reward, next_state)
            state = next_state
            if state == n_states - 1:
                break

    print("Q-Table:")
    print(q_learning.q_table)
