import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters initialization
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)  # Forget gate weights
        self.bf = np.zeros((hidden_size, 1))  # Forget gate bias
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)  # Input gate weights
        self.bi = np.zeros((hidden_size, 1))  # Input gate bias
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)  # Output gate weights
        self.bo = np.zeros((hidden_size, 1))  # Output gate bias
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)  # Cell state weights
        self.bc = np.zeros((hidden_size, 1))  # Cell state bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, prev_hidden_state, prev_cell_state):
        # Concatenate input and previous hidden state
        concat_input = np.vstack((prev_hidden_state, x))

        # Forget gate
        f = self.sigmoid(np.dot(self.Wf, concat_input) + self.bf)

        # Input gate
        i = self.sigmoid(np.dot(self.Wi, concat_input) + self.bi)

        # Output gate
        o = self.sigmoid(np.dot(self.Wo, concat_input) + self.bo)

        # Candidate cell state
        c_hat = self.tanh(np.dot(self.Wc, concat_input) + self.bc)

        # Update cell state
        cell_state = f * prev_cell_state + i * c_hat

        # Update hidden state
        hidden_state = o * self.tanh(cell_state)

        return hidden_state, cell_state

# Example usage
input_size = 3
hidden_size = 4
lstm_cell = LSTM(input_size, hidden_size)

# Input sequence
x_t = np.random.randn(input_size, 1)
# Initial hidden state and cell state
h_t_minus_1 = np.random.randn(hidden_size, 1)
c_t_minus_1 = np.random.randn(hidden_size, 1)

# Forward pass through LSTM cell
h_t, c_t = lstm_cell.forward(x_t, h_t_minus_1, c_t_minus_1)

print("Output hidden state h_t:", h_t)
print("Output cell state c_t:", c_t)
