import numpy as np

class SVMClassifier:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
class SVMRegressor:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, epsilon=0.1):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.epsilon = epsilon
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = np.abs(y[idx] - (np.dot(x_i, self.w) + self.b)) <= self.epsilon
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx] - np.sign(y[idx] - (np.dot(x_i, self.w) + self.b)) * self.epsilon))
                    self.b -= self.learning_rate * (y[idx] - np.sign(y[idx] - (np.dot(x_i, self.w) + self.b)) * self.epsilon)

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# Example usage:
if __name__ == "__main__":
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])  # Labels should be 0 or 1
    
    # Create and train the classifier
    classifier = SVMClassifier(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    classifier.fit(X, y)
    
    # Make predictions
    predictions = classifier.predict(X)
    print("Classification Predictions:", predictions)
