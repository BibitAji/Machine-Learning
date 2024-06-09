import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_encoder = np.random.randn(self.input_dim, self.hidden_dim)
        self.bias_encoder = np.zeros((1, self.hidden_dim))
        self.weights_decoder = np.random.randn(self.hidden_dim, self.input_dim)
        self.bias_decoder = np.zeros((1, self.input_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X):
        for epoch in range(self.epochs):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_encoder) + self.bias_encoder
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_decoder) + self.bias_decoder
            output_layer_output = self.sigmoid(output_layer_input)

            # Compute loss (Mean Squared Error)
            loss = np.mean((X - output_layer_output) ** 2)

            # Backpropagation
            error = X - output_layer_output
            d_output = error * self.sigmoid_derivative(output_layer_output)

            error_hidden_layer = d_output.dot(self.weights_decoder.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_decoder += hidden_layer_output.T.dot(d_output) * self.learning_rate
            self.bias_decoder += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
            self.weights_encoder += X.T.dot(d_hidden_layer) * self.learning_rate
            self.bias_encoder += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss:.6f}')

    def encode(self, X):
        hidden_layer_input = np.dot(X, self.weights_encoder) + self.bias_encoder
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        return hidden_layer_output

    def decode(self, hidden_representation):
        output_layer_input = np.dot(hidden_representation, self.weights_decoder) + self.bias_decoder
        output_layer_output = self.sigmoid(output_layer_input)
        return output_layer_output

    def reconstruct(self, X):
        hidden_representation = self.encode(X)
        reconstructed_X = self.decode(hidden_representation)
        return reconstructed_X

# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ])

    # Create and train autoencoder
    autoencoder = Autoencoder(input_dim=4, hidden_dim=2, learning_rate=0.1, epochs=1000)
    autoencoder.fit(X)

    # Encode and decode the sample data
    encoded_data = autoencoder.encode(X)
    decoded_data = autoencoder.reconstruct(X)

    print("Original Data:\n", X)
    print("Encoded Data:\n", encoded_data)
    print("Reconstructed Data:\n", decoded_data)
